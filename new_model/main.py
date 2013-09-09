#!/usr/bin/env python

from __future__ import division

import sys
sys.path.insert(1,"..")
from automain import automain
from argparse import ArgumentParser
from model import get_session_factory
# from observation import Observation
from lmk_rel_observations import (LandmarkObservation, RelationObservation,
                                  Utterance)

from semantics.run import read_scenes, construct_training_scene
import utils
from utils import logger
# from planar import Vec2
# from semantics.landmark import Landmark, Color, ObjectClass
# from semantics.representation import PointRepresentation
from myrandom import random
choice = random.choice
random = random.random

from Orange.feature import Discrete, Continuous, Descriptor
from Orange.data.continuization import DomainContinuizer
from Orange.data import Value
from Orange.data import Domain, Instance, Table
from Orange.classification.tree import TreeLearner, TreeClassifier, Node
from Orange.classification.knn import kNNLearner
from Orange.classification.bayes import NaiveLearner
from Orange.ensemble.forest import RandomForestLearner
from Orange.regression.earth import EarthLearner, plot_evimp
from Orange.evaluation.testing import cross_validation
from Orange.evaluation.scoring import (Sensitivity, Specificity,
                                       Precision, Recall, F1, Falpha)
from orngTree import printTree

from landmark_from_description import get_landmark_prob
from scipy.optimize import *
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

from semantics.teacher import Teacher, Semantic

from multiprocessing import Pipe, Process
from socket import setdefaulttimeout
setdefaulttimeout(None)
from itertools import izip

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

import shelve
import time
import traceback

import IPython

from collections import defaultdict

import bernoulli_regression_tree as brt

# from student_trees import trees as regression_trees

def spawn(f):
    def fun(ppipe, cpipe,x):
        ppipe.close()
        cpipe.send(f(x))
        cpipe.close()
    return fun

def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(p,c,x)) for x,(p,c) in izip(X,pipe)]
    [p.start() for p in proc]
    ret = [p.recv() for (p,c) in pipe]
    [p.join() for p in proc]
    return ret

def perfect_train_one(args):
    speaker, scene, session_factory = args
    utils.scene.set_scene(scene,speaker)
    table = scene.landmarks['table'].representation.rect
    obj_lmks = [lmk for lmk in scene.landmarks.values() if lmk.name != 'table']
    t_min = table.min_point
    # t_max = table.max_point
    t_w = table.width
    t_h = table.height

    xloc,yloc = random()*t_w+t_min.x, random()*t_h+t_min.y
    # trajector = Landmark( 'point', PointRepresentation(Vec2(xloc,yloc)), None, Landmark.POINT)
    trajector = choice(obj_lmks)
    sentence, relation, landmark = speaker.describe(trajector, scene, False, 1)

    #TODO make Utterance

    LandmarkObservation.make_observations(session=session_factory(), 
                                          speaker=speaker,
                                          landmark=landmark,
                                          landmark_prior=1.0,
                                          utterance=sentence)

    RelationObservation.make_observations(session=session_factory(), 
                                          speaker=speaker,
                                          landmark=landmark,
                                          landmark_prior=1.0,
                                          trajector=trajector,
                                          utterance=sentence,
                                          positive=True)

    for obj_lmk in obj_lmks:
        if obj_lmk != trajector:
            RelationObservation.make_observations(session=session_factory(), 
                                                  speaker=speaker,
                                                  landmark=landmark,
                                                  landmark_prior=1.0,
                                                  trajector=obj_lmk,
                                                  utterance=sentence,
                                                  positive=False)

def perfect_train(session_factory, iterations, scene_descs):
    
    if scene_descs:
        per_scene = int(np.ceil(iterations / len(scene_descs)))

    j = 0
    for i in range(iterations):
        print 'sentence', i

        if scene_descs:
            if (i%per_scene) == 0:
                scene, speaker, _ = scene_descs[j]
                j += 1
        else:
            if (i % 50) == 0:
                scene, speaker = construct_training_scene(True)

        perfect_train_one((speaker,scene,session_factory))


# For classification tree
trajector_distance    = Continuous("trajector_distance")
trajector_surface     = Discrete("trajector_surface", values=["False","True","None"])
trajector_contained   = Discrete("trajector_contained", values=["False","True","None"])
trajector_in_front_of = Discrete("trajector_in_front_of", values=["False","True","None"])
trajector_behind      = Discrete("trajector_behind", values=["False","True","None"])
trajector_left_of     = Discrete("trajector_left_of", values=["False","True","None"])
trajector_right_of    = Discrete("trajector_right_of", values=["False","True","None"])

positive              = Discrete("positive", values=["False","True"])
applicability         = Continuous("true_relation_score")
relation_phrase       = Discrete("relation_phrase")

features = [
            trajector_distance
            ,trajector_surface
            ,trajector_contained
            ,trajector_in_front_of
            ,trajector_behind
            ,trajector_left_of
            ,trajector_right_of
            ,positive
            ]

features2 = [
            trajector_distance
            ,trajector_surface
            ,trajector_contained
            ,trajector_in_front_of
            ,trajector_behind
            ,trajector_left_of
            ,trajector_right_of
            ,applicability
            ]

features3 = [
            trajector_distance
            ,trajector_surface
            ,trajector_contained
            ,trajector_in_front_of
            ,trajector_behind
            ,trajector_left_of
            ,trajector_right_of
            ,relation_phrase
            ]

feature_columns = [
                RelationObservation.trajector_distance,
                RelationObservation.trajector_surface,
                RelationObservation.trajector_contained,
                RelationObservation.trajector_in_front_of,
                RelationObservation.trajector_behind,
                RelationObservation.trajector_left_of,
                RelationObservation.trajector_right_of,
                RelationObservation.positive,
                RelationObservation.landmark_prior,
                RelationObservation.true_relation_applicability
                ]

instance_weight = Descriptor.new_meta_id()
true_applicability = Descriptor.new_meta_id()
true_score = Descriptor.new_meta_id()
def make_orange_table(session_factory, relation_phrases, features=features):
    domain = Domain(features)
    domain.add_meta(instance_weight, Continuous("weight"))
    domain.add_meta(true_applicability, Continuous("true_applicability"))
    domain.add_meta(true_score, Continuous("true_score"))

    session = session_factory()
    instances = []
    def add_instances(q, positive_only=False):
        for row in q:
            if row.landmark_prior is None:
                continue
            if positive_only and not row.positive:
                continue
            instance_features = []
            for feature in features:
                value = getattr(row,feature.name)
                if isinstance(feature, Discrete):
                    value = str(value)
                    feature.add_value(value)
                instance_features.append(value)
            instance = Instance(domain,instance_features)
            # instance.set_class(row.positive)
            instance[instance_weight] = row.landmark_prior
            instance[true_applicability] = row.true_relation_applicability
            instance[true_score] = row.true_relation_score
            instances.append(instance)

    if relation_phrases is None:
        q = session.query(RelationObservation)
        add_instances(q, positive_only=True)
    else:
        for relation_phrase in relation_phrases:
            q = session.query(RelationObservation).filter(
                RelationObservation.relation_phrase == relation_phrase)
            add_instances(q)



    data = Table(domain, instances)

    # for row in data[:10]:
    #   print row, row.get_class()

    data0 = continuizer = None
    # continuizer = DomainContinuizer()
    # continuizer.multinomial_treatment = continuizer.AsOrdinal
    # domain0 = continuizer(data)
    # domain0.add_meta(instance_weight, Continuous("weight"))
    # domain0.add_meta(true_applicability, Continuous("true_applicability"))
    # domain0.add_meta(true_score, Continuous("true_score"))
    # data0 = data.translate(domain0)
    # for instance1, instance2 in zip(data,data0):
    #   for id,value in instance1.get_metas().items():
    #       instance2.set_meta(id,value.value)

    return data, data0, continuizer, instance_weight



def make_decision_tree(data_table, TLkwargs={}):


    try:
        relation_tree = TreeLearner(data_table, 
                                    weightID='weight',
                                    **TLkwargs)
    except Exception as e:
        traceback.print_exc()
        for row in data_table:
            logger( row )
        raise e

    # for instance in relation_table:
    #   # if instance[-1] == 'True':
    #   print relation_tree(instance), instance
    # print
    # for i in range(2,5):
    #   print i,'fold cross-validation:'
    #   res = cross_validation([TreeLearner], relation_table, folds=i)
    #   print '  Precision:',Precision(res)
    #   print '  Recall:',Recall(res)
    #   print '  F1:',F1(res)
    #   print '  Falpha:',Falpha(res)
    #   print '  Sensitivity:',Sensitivity(res)
    #   print '  Specificity:',Specificity(res)

    return relation_tree

def test_decision_tree(instances, tree):
    domain = Domain(features)
    relation_table = Table(domain, instances)
    return [tree(instance) for instance in relation_table]


def test_landmark_phrases(session_factory, scene_descs):

    session = session_factory()
    scene, speaker, image = scene_descs[0]

    utils.scene.set_scene(scene,speaker)
    # table = scene.landmarks['table'].representation.rect
    # obj_lmks = [lmk for lmk in scene.landmarks.values() if lmk.name != 'table']
    all_lmks = scene.landmarks.values() + scene.landmarks['table'].representation.landmarks.values()
    plt.ion()
    fig = plt.imshow(image)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()

    landmark_description = raw_input('Landmark: ')
    while landmark_description != 'exit':

        try:
            scene_num = int(landmark_description)
            if scene_num in range(5):
                scene, speaker, image = scene_descs[scene_num]

                utils.scene.set_scene(scene,speaker)
                # table = scene.landmarks['table'].representation.rect
                # obj_lmks = [lmk for lmk in scene.landmarks.values() if lmk.name != 'table']
                all_lmks = scene.landmarks.values() + scene.landmarks['table'].representation.landmarks.values()
                plt.ion()
                fig = plt.imshow(image)
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.show()
                landmark_description = raw_input('Landmark: ')
                continue
        except:
            pass

        utterance = "behind "+landmark_description
        lmk_probs = []
        for lmk in all_lmks:#obj_lmks:
            lmk_probs.append(get_landmark_prob(session, utterance, 
                                             speaker, lmk))
        lmk_probs = np.array(lmk_probs)
        lmk_probs/=lmk_probs.sum()

        for lmk,prob in zip(all_lmks,lmk_probs):#obj_lmks,obj_lmk_probs):
            print lmk, prob
        landmark_description = raw_input('Landmark: ')

def test_teacher(scene_descs):

    teacher = Teacher((0,0))

    command = '0'
    while command != 'exit':

        try:
            scene_num = int(command)
            if scene_num in range(5):
                scene, speaker, image = scene_descs[scene_num]

                utils.scene.set_scene(scene,speaker)
                teacher.set_location(speaker.location)
                # table = scene.landmarks['table'].representation.rect
                obj_lmks = [lmk for lmk in scene.landmarks.values() 
                            if lmk.name != 'table']
                # all_lmks = scene.landmarks.values() + scene.landmarks['table'].representation.landmarks.values()
                plt.ion()
                fig = plt.imshow(image)
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.show()
                command = raw_input('Command: ')
                continue
        except:

            trajector = choice(obj_lmks)
            print trajector
            print teacher.describe(scene, trajector)

            command = raw_input('Command: ')

def regular_train_landmark_loop(args):
    global session_factory
    scene_desc, iterations, cheat, test_only, TLkwargs = args
    scene, speaker = scene_desc
    teacher = Teacher(location=speaker.location)
    utils.scene.set_scene(scene,speaker)
    teacher.set_location(speaker.location)

    # object_landmarks = teacher.get_object_landmarks(scene)
    all_landmarks = teacher.get_landmarks(scene)

    answers = []
    for i in range(iterations):

        trace = 'Iteration %i\n' % i
        trajector = choice(all_landmarks)
        utterance = teacher.describe_trajector(scene, trajector)

        trace += 'Teacher trajector: %s\n' % str(trajector) 
        trace += '        utterance: %s\n' % utterance 

        utterance = 'behind ' + utterance
        lmk_probs = []
        for lmk in all_landmarks:
            lmk_probs.append(get_landmark_prob(session_factory(), 
                                               utterance, 
                                               teacher, 
                                               lmk))
        lmk_probs = np.array(lmk_probs)
        lmk_probs/=lmk_probs.sum()

        landmark_probs = sorted(zip(lmk_probs,all_landmarks),reverse=True)
        trace += 'Student landmark guesses:\n'
        for prob, lmk in landmark_probs[:10]:
            trace += '  Landmark: %s, Prob: %f\n' % (str(lmk), prob)

        _, trajector_guess = landmark_probs[0]
        trace += '%sStudent trajector guess: \n  %s\n  %s%s\n' % (
                utils.printcolors.OKGREEN,
                str(trajector_guess),
                trajector_guess == trajector,
                utils.printcolors.ENDC)
        answers.append( (str(trajector),trajector_guess==trajector) )

        LandmarkObservation.make_observations(session=session_factory(), 
                              speaker=teacher,
                              landmark=trajector,
                              landmark_prior=1.0,
                              utterance=utterance)
        logger( trace )
    return answers

# relation decision_trees = {}

def regular_train_relation_loop(args):
    global session_factory
    # global relation_decision_trees
    scene_desc, iterations, cheat, test_only, train_only, TLkwargs = args
    scene, speaker = scene_desc
    teacher = Teacher(location=speaker.location)
    utils.scene.set_scene(scene,speaker)
    teacher.set_location(speaker.location)

    object_landmarks = teacher.get_object_landmarks(scene)
    all_landmarks = teacher.get_landmarks(scene)

    answers = []
    for i in range(iterations):

        trace = 'Iteration %i\n' % i

        trajector = choice(object_landmarks)
        meaning = teacher.sample_meaning(scene, trajector)
        utterance = teacher.describe_meaning(meaning)

        trace += 'Teacher trajector: %s\n' % str(trajector) 
        trace += '        utterance: %s\n' % utterance 

        if cheat:
            landmark_guess = meaning.landmark
        else:
            lmk_probs = []
            for lmk in all_landmarks:
                lmk_probs.append(get_landmark_prob(session_factory(), 
                                                   utterance, 
                                                   teacher, 
                                                   lmk))
            lmk_probs = np.array(lmk_probs)
            lmk_probs/=lmk_probs.sum()

            landmark_probs = sorted(zip(lmk_probs,all_landmarks),reverse=True)
            trace += 'Student landmark guesses:\n'
            for prob, lmk in landmark_probs[:10]:
                trace += '  Landmark: %s, Prob: %f\n' % (str(lmk), prob)
            
            _, landmark_guess = landmark_probs[0]

        if landmark_guess in object_landmarks:
            gi = object_landmarks.index(landmark_guess)
            trajector_candidates = object_landmarks[:gi]+\
                object_landmarks[gi+1:]
        else:
            trajector_candidates = object_landmarks
        trace += 'Student landmark guess: %s\n' % str(landmark_guess)
        linguistic_features = RelationObservation.\
                                extract_linguistic_features(session_factory(), 
                                                            utterance)
        trace += linguistic_features['relation_phrase'] + '\n'
        if train_only:
            answers.append(False)
        else:
            try:
                relation_table = make_orange_table(session_factory,
                    [linguistic_features['relation_phrase']])[0]
            except Exception as e:
                answers.append(  ( (str(meaning.relation),
                                    str(meaning.landmark),
                                    str(landmark_guess)),
                                   False )  )
                trace += str(e)
            else:
                relation_tree = make_decision_tree(relation_table,
                                            TLkwargs=TLkwargs)
                instances = []


                for trajector_candidate in trajector_candidates:
                    semantic_features = RelationObservation.\
                                        extract_semantic_features(teacher, 
                                                                  landmark_guess, 
                                                                  trajector_candidate)
                    # semantic_features['positive'] = False
                    instance = []
                    for feature in features:
                        if feature.name != 'positive':
                            value = semantic_features[feature.name]
                            if isinstance(feature, Discrete):
                                value = str(value)
                                feature.add_value(value)
                            instance.append(value)
                    instance.append(None)
                    instances.append(instance)

                for candidate, instance in zip(trajector_candidates,instances):
                    trace += str(candidate) + ' '
                    trace += str(instance) + ' '
                    trace += str(teacher.get_applicability(scene,
                                                        meaning.relation,
                                                        landmark_guess,
                                                        candidate)) + ' '
                    trace += str(teacher.get_score(scene,
                                                meaning.relation,
                                                landmark_guess,
                                                candidate))
                    trace += '\n'
                results = test_decision_tree(instances, relation_tree)

                # feature_array = np.rec.fromrecords(instances, 
                #                        names=[f.name for f in features])
                # student_applicabilities = \
                #     regression_trees[linguistic_features['relation_phrase']](
                #         feature_array)
                # max_app = max(student_applicabilities)
                # results = (np.array(student_applicabilities)==max_app).astype(float)

                trace += 'Student trajector estimations:\n'
                for result, trajector_candidate in zip(results,trajector_candidates):
                    trace += '  Object: %s, %s? %s\n' %(str(trajector_candidate),
                                            linguistic_features['relation_phrase'],
                                            result)

                results = np.array([1.0 if result.native()=="True" else 0.0 
                                    for result in results])
                if results.sum() > 0:
                    results /= results.sum()
                    logger(results)
                    # trajector_guess = utils.categorical_sample(trajector_candidates,
                    #                                            results)[0]
                    sorted_results = sorted(zip(results,object_landmarks),
                                           reverse=True)
                    trace += str(sorted_results)
                    trajector_guess = sorted_results[0][1]
                    trace += '%sStudent trajector guess: \n  %s\n  %s%s\n' % (
                        utils.printcolors.OKGREEN,
                        str(trajector_guess),
                        trajector_guess == trajector,
                        utils.printcolors.ENDC)

                    answers.append(  ( (str(meaning.relation),
                                        str(meaning.landmark),
                                        str(landmark_guess) ),
                                      trajector_guess == trajector)  )
                else:
                    answers.append(False)

        session=session_factory()
        utterance_entry = Utterance(text=utterance)
        session.add(utterance_entry)
        session.commit()

        if not test_only:
            correct_trajector = trajector # Teacher tells us answer
            if cheat:
                for obj_lmk in object_landmarks:
                    if meaning.landmark != obj_lmk:
                        RelationObservation.\
                        make_observations(session=session_factory(), 
                                          speaker=teacher,
                                          landmark=landmark_guess,
                                          landmark_prior=1.0,
                                          trajector=obj_lmk,
                                          utterance=utterance_entry,
                                          positive=obj_lmk == correct_trajector,
                                          true_landmark=meaning.landmark,
                                          true_relation_applicability=
                                            teacher.get_applicability(scene,
                                                                meaning.relation,
                                                                meaning.landmark,
                                                                obj_lmk),
                                          true_relation_score=
                                            teacher.get_score(scene,
                                                                meaning.relation,
                                                                meaning.landmark,
                                                                obj_lmk)
                                          )

            else:
                for obj_lmk in object_landmarks:
                    # Not certain about landmark, so train on top 5
                    for prob, landmark in landmark_probs[:5]:
                        if landmark != obj_lmk:
                            RelationObservation.\
                            make_observations(session=session_factory(), 
                                          speaker=teacher,
                                          landmark=landmark,
                                          landmark_prior=prob,
                                          trajector=obj_lmk,
                                          utterance=utterance_entry,
                                          positive=obj_lmk==correct_trajector,
                                          true_landmark=meaning.landmark,
                                          true_relation_applicability=
                                            teacher.get_applicability(scene,
                                                                meaning.relation,
                                                                landmark,
                                                                obj_lmk),
                                          true_relation_score=
                                            teacher.get_score(scene,
                                                                meaning.relation,
                                                                landmark,
                                                                obj_lmk)
                                        )

        # Don't observe landmark, since you don't know which is right
        trace += '\n'
        logger( trace )
    return answers

def interleave(*args):
    for idx in range(0, max(len(arg) for arg in args)):
        for arg in args:
            try:
                yield arg[idx]
            except IndexError:
                continue

def running_avg(arr,window):
        return  [np.mean(arr[:i]) for i in range(1,window)] + [np.mean(arr[i-window:i]) for i in range(window,len(arr))]

def regular_train(session_factory1, scene_descs, iterations, 
                  save_file=None, just_landmark=False, 
                  parallel=True, cheat=False, test_only=False, train_only=False,
                  TLkwargs={}):

    global session_factory
    session_factory = session_factory1
    per_scene = int(np.floor(iterations/len(scene_descs)))

    args = zip(scene_descs,
               [per_scene]*len(scene_descs),
               [cheat]*len(scene_descs),
               [test_only]*len(scene_descs),
               [train_only]*len(scene_descs),
               [TLkwargs]*len(scene_descs))

    if just_landmark:
        if parallel:
            results = parmap(regular_train_landmark_loop, args)
            answers = list(interleave(*results))
        else:
            results = map(regular_train_landmark_loop, args)
            answers = []
            for result in results:
                answers.extend(result)

        if save_file is not None:
            f = shelve.open(save_file)
            if 'landmark_answers' in f:
                f['landmark_answers'] = f['landmark_answers'] + answers
            else:
                f['landmark_answers'] = answers
            f.close()
    else:

        if parallel:
            results = parmap(regular_train_relation_loop, args)
            answers = list(interleave(*results))
        else:
            results = map(regular_train_relation_loop, args)
            answers = []
            for result in results:
                answers.extend(result)

        if save_file is not None:
            f = shelve.open(save_file)
            if 'relation_answers' in f:
                f['relation_answers'] = f['relation_answers'] + answers
            else:
                f['relation_answers'] = answers
            f.close()

    print len(results)
    return answers

def random_scene_training(session_factory1, iterations, per_scene, 
                          save_file=None, just_landmark=False, parallel=True, 
                          cheat=False, test_only=False, train_only=False,
                          TLkwargs={}):

    num_scenes = int(np.ceil(iterations/per_scene))
    scene_descs = [construct_training_scene(random=True) 
                   for _ in range(num_scenes)]
    regular_train(session_factory1=session_factory1,
                  scene_descs=scene_descs,
                  iterations=iterations,
                  save_file=save_file,
                  just_landmark=just_landmark,
                  parallel=parallel,
                  cheat=cheat,
                  test_only=test_only,
                  train_only=train_only,
                  TLkwargs=TLkwargs)

    

def cross_validate(db_url, scene_descs, filename, cheat):
    import subprocess
    TLkwargs = {'maxDepth':3}
    accuracies = []
    for i in range(len(scene_descs)):

        subprocess.call(['cp','landmark_relation_trained_1000.db','test.db'])
        session_factory = get_session_factory(db_url, echo=False)

        test_scene = scene_descs[i]
        train_scenes = scene_descs[:i]+scene_descs[i+1:]

        regular_train(session_factory, 
                  [(scene, speaker) for scene,speaker,_ in train_scenes], 
                  600, 
                  save_file=filename,
                  just_landmark=False,
                  cheat=cheat,
                  TLkwargs=TLkwargs)

        results = regular_train(session_factory, 
                  [(scene, speaker) for scene,speaker,_ in [test_scene]], 
                  150, 
                  save_file=filename,
                  just_landmark=False,
                  cheat=cheat,
                  test_only=True,
                  TLkwargs=TLkwargs)

        _, results = zip(*results)
        accuracies.append(np.mean(results))
    print accuracies



def tree_summary0(node, level):
    if not node:
        return None
    if node.branch_selector:
        node_desc = node.branch_selector.class_var.name
        node_cont = node.distribution
        return [[node_desc, description, node_cont, 
                 tree_summary0(branch, level+1)] 
                for branch, description in zip(node.branches, 
                                               node.branch_descriptions)]
    else:
        node_cont = node.distribution
        major_class = node.node_classifier.default_value
        return (major_class, node_cont)

def tree_summary(x):
    if isinstance(x, TreeClassifier):
        return tree_summary0(x.tree, 0)
    elif isinstance(x, Node):
        return tree_summary0(x, 0)
    else:
        raise TypeError, "invalid parameter"

# def compare_decision_trees(session_factory):    
#     session = session_factory()
#     q = session.query(RelationObservation.relation_phrase)
#     relation_phrases = set(zip(*q)[0])
#     trees = [make_decision_tree(session_factory,
#                                 # relation_phrase,
#                                 TLkwargs={'maxDepth':3})[0]
#              for relation_phrase in relation_phrases]
#     trees
#     tree_summaries = [tree_summary(tree) for tree in trees]

#     for phrase, summary in zip(relation_phrases,tree_summaries):
#         print phrase, summary


def make_some_histograms(session_factory):
    session = session_factory()
    q = session.query(RelationObservation.relation_phrase)
    relation_phrases = set(zip(*q)[0])

    # IPython.embed()
    plt.ion()
    # for relation_phrase in relation_phrases:
    # with 'behind' as relation_phrase:
    relation_phrase = 'near to'
    q = session.query(RelationObservation).\
          filter(RelationObservation.positive==True)
    total_feature_values = [[getattr(row,feature.name) 
                       for feature in features]
                      for row in q]
    total_priors = [row.landmark_prior for row in q]
    total_feature_values = zip(*total_feature_values)
    q = q.filter(RelationObservation.relation_phrase==relation_phrase)
    feature_values = [[getattr(row,feature.name) 
                       for feature in features]
                      for row in q]
    priors = [row.landmark_prior for row in q]
    feature_values = zip(*feature_values)
    for feature, values, total_values \
            in zip(features, feature_values, total_feature_values):
        plt.figure()
        if hasattr(feature,'values'):
            ind = np.arange(len(feature.values))
            width = 0.35
            counter = defaultdict(float)
            counter_sum = 0
            for value, prior in zip(values,priors):
                counter[str(value)]+=prior 
                counter_sum += prior
            heights = [counter[value]/counter_sum for value in feature.values]
            print zip(feature.values, heights)
            plt.bar(ind,heights,width,facecolor='red', alpha=0.75, label='Posterior')

            total_counter = defaultdict(float)
            total_counter_sum = 0
            for value, prior in zip(total_values,total_priors):
                total_counter[str(value)]+=prior 
                total_counter_sum+=prior
            total_heights = [total_counter[value]/total_counter_sum for value in feature.values]
            print zip(feature.values, total_heights)
            plt.bar(ind+width,total_heights,width,facecolor='blue', alpha=0.75, label='Prior')
            plt.gca().set_xticks(ind+width)
            plt.gca().set_xticklabels( feature.values )
        else:
            # plt.subplot(121)
            plt.hist(values,bins=10,weights=priors,normed=1,facecolor='red', alpha=0.75, label='Posterior')
            # plt.subplot(122)
            plt.hist(total_values,bins=10,weights=total_priors,normed=1,facecolor='blue', alpha=0.75, label='Prior')
        plt.title(feature.name)
        plt.legend().draggable(True)
    raw_input()
    # IPython.embed()

def average(xs, weights):
    return np.dot(xs,weights)/weights.sum()

def irregular_moving_average(xs,ys,weights):
    xs = np.array(xs)
    ys = np.array(ys)
    weights = np.array(weights)
    minx, maxx = xs.min(),xs.max()
    half_window_size = float(maxx-minx)/10.0/2
    avgs = []
    for x in xs:
        ind = np.where(np.logical_and(xs>x-half_window_size,
                                      xs<x+half_window_size))
        avgs.append(average(ys[ind],weights=weights[ind]))
    return np.array(avgs)

def irregular_moving_average2(xs,ys,weights,window_size=200):
    xs = np.array(xs)
    ys = np.array(ys)
    weights = np.array(weights)
    sorted_ind = np.argsort(xs)
    end = len(xs)-1
    avgs = np.empty(xs.shape)
    for i in range(len(xs)):
        l = max(0,i-window_size)
        h = min(i+window_size,end)
        window = sorted_ind[l:h]
        avg = average(ys[window],weights=weights[window])
        avgs[sorted_ind[i]] = avg
    return np.array(avgs)

def distance_graphs(session_factory, relation_phrases):

    # all_data,_,_,_ = make_orange_table(session_factory=session_factory,
    #                                  relation_phrases=None,
    #                                  features=features3)
    # all_knn = kNNLearner(k=50)(all_data)

    data, _,_, weightID = make_orange_table(session_factory=session_factory,
                                            relation_phrases=relation_phrases)

    # weight_range = (0.0,1.0)
    # data = data.filter(weight=weight_range)
    table = data
    print '# datapoints:',len(table)

    # tree = make_decision_tree(table,
    #                         TLkwargs={'maxDepth':3})
    # knn = kNNLearner(k=10)(table)
    # knn.weight_ID = weightID
    # bayes = NaiveLearner(table, weightID=weightID)
    forest = RandomForestLearner(trees=500,
                                 attributes=7,
                                 base_learner=TreeLearner(min_instances=5)
                                 )(table, weight=weightID)
    data2,_,_,_ = make_orange_table(session_factory=session_factory,
                                    relation_phrases=relation_phrases,
                                    features=features2)
    # feature_array,weights = table.to_numpy('a/w',weightID=weightID)
    # values = feature_array[:,0],
    
    # data2 = data2.filter(weight=weight_range)

    # knn_smoothed = []
    for row1, row2 in zip(data,data2):
        value = forest(row1,1)[1]
        # value = 0
        # values = all_knn(row1,1)
        # for phrase in relation_phrases:
            # value += values[relation_phrase.values.index(phrase)]
        row2[-1] = Value(applicability,value)
        # knn_smoothed.append(value)
    # knn_smoothed = np.array(knn_smoothed,dtype=np.float64)
    earth = EarthLearner(degree=2,terms=None)(data2, weight_id=weightID)
    # IPython.embed()
    # exit()
    true_applicabilities = []
    true_scores          = []
    weights              = []
    positives            = []
    # for i, row in enumerate(table):

    # tree_probabilities   = []
    # knn_probabilities    = []
    # bayes_probabilities  = []
    forest_probabilities = []
    earth_probabilities  = []
    feature_values = [[] for _ in features]
    for i, row in enumerate(table):
        # tree_probabilities.append( tree(row,1)[1] )
        # knn_probabilities.append( knn(row,1)[1] )
        # bayes_probabilities.append( bayes(row,1)[1] )
        forest_probabilities.append( forest(row,1)[1] )
        earth_probabilities.append( earth(row) )
        true_applicabilities.append( row['true_applicability'].native() )
        true_scores.append( row['true_score'].native() )
        weights.append( row['weight'].native() )
        positives.append( row[len(row)-1].native()=='True' )
        for j in range(len(features)):
            feature_values[j].append( row[j].native() )
    feature_array = table.to_numpy()[0].astype(np.float64)
    feature, values = features[0], feature_values[0]

    moving_average = irregular_moving_average(values,positives,weights)

    # tree_probabilities = np.array(tree_probabilities,dtype=np.float64)
    # knn_probabilities = np.array(knn_probabilities,dtype=np.float64)
    # bayes_probabilities = np.array(bayes_probabilities,dtype=np.float64)
    forest_probabilities = np.array(forest_probabilities,dtype=np.float64)
    earth_probabilities = np.array(earth_probabilities,dtype=np.float64)
    true_applicabilities = np.array(true_applicabilities,dtype=np.float64)
    true_scores = np.array(true_scores,dtype=np.float64)
    weights = np.array(weights,dtype=np.float64)
    positives = np.array(positives,dtype=np.float64)
    moving_average = np.array(moving_average,dtype=np.float64)


    colors = ['g' if p else 'r' for p in positives]
    # ensemble_probabilities = np.array([np.mean(tup) for tup in zip(
    #                                                     # tree_probabilities
    #                                                     knn_probabilities
    #                                                     ,moving_average
    #                                                     )],dtype=np.float64)

    # print feature_array[:,0]
    # print feature_array[:,1]
    # print feature_array[:,2]

    def reverse_norm_cdf(xdata, params):
        return (1-norm.cdf(xdata[:,0],params[0],params[1]))

    def reverse_norm_cdf_tree(xdata, params):
        return (1-xdata[:,1])*(1-xdata[:,2])*\
               (1-norm.cdf(xdata[:,0],params[0],params[1]))

    def errfunc(params, func, xdata, ydata, weights):
        return np.array(ydata - func(xdata,params))*np.array(weights)

    def sserrfunc(params, func, xdata, ydata, weights):
        return np.sum(errfunc(params,func,xdata,ydata,weights)**2)

    func = reverse_norm_cdf_tree

    # def errfunc2(params, xdata, ydata, weights):
    #   return np.array(ydata - )*np.array(weights)

    # p0 = {'loc':0,'scale':1}
    p0 = [1,2,1,1]
    minimize = leastsq
    errfunc = errfunc
    p1 = minimize(errfunc,p0,args=(func,feature_array,positives,weights))[0]
    # p2 = minimize(errfunc,p0,args=(func,feature_array,moving_average,weights))[0]
    p3 = minimize(errfunc,p0,args=(func,feature_array,forest_probabilities,weights))[0]
    print 'p1',p1
    # print 'p2',p2
    print 'p3',p3
    # p = [0.15,0.05,1,1]
    regression_probabilities1 = func(feature_array, p1)
    # regression_probabilities2 = func(feature_array, p2)
    regression_probabilities3 = func(feature_array, p3)
    vmin, vmax = min(values), max(values)
    regression_line_xs = np.linspace(vmin,vmax,50).reshape(50,1)
    func_xs = np.append(regression_line_xs, np.zeros((50,2)),1)
    regression_line_1_ys = func(func_xs,p1)
    # regression_line_2_ys = func(func_xs,p2)
    regression_line_3_ys = func(func_xs,p3)

    plt.ion()
    # for feature, values in zip(features,feature_values):
    feature, values = features[0], feature_values[0]
    s = 50*(4**np.array(weights))
    x_label = "Distance between Trajector and Landmark"
    # plt.figure()
    # plt.subplot(191).scatter(values,knn_probabilities,s=s,c=colors)
    # plt.ylim([-0.1,1.1])
    # plt.title('kNN Probability')
    # plt.subplot(192).scatter(values,bayes_probabilities,s=s,c=colors)
    # plt.ylim([-0.1,1.1])
    # plt.title('Bayes Probability')
    # plt.subplot(191)

    plt.figure()
    plt.scatter(values,earth_probabilities,s=s,c=colors)
    plt.ylim([-0.1,1.1])
    plt.xlabel(x_label)
    plt.ylabel('Estimated applicability of ' + relation_phrases[0])
    plt.title('MARS Probability')

    # plt.figure()
    # plt.scatter(values,knn_smoothed,s=s,c=colors)
    # plt.ylim([-0.1,1.1])
    # plt.title('kNN Smoothed')

    # plt.subplot(192)
    plt.figure()
    plt.scatter(values,forest_probabilities,s=s,c=colors)
    plt.ylim([-0.1,1.1])
    plt.xlabel(x_label)
    plt.ylabel('Estimated applicability of ' + relation_phrases[0])
    plt.title('Forest Probability')

    # plt.subplot(193)
    # plt.figure()
    # plt.scatter(values,tree_probabilities,s=s,c=colors)
    # plt.ylim([-0.1,1.1])
    # plt.title('Tree Probability')

    # plt.subplot(194)
    plt.figure()
    plt.scatter(values,regression_probabilities3,s=s,c=colors)
    plt.plot(regression_line_xs,regression_line_3_ys,'-')
    plt.ylim([-0.1,1.1])
    plt.xlabel(x_label)
    plt.ylabel('Estimated applicability of ' + relation_phrases[0])
    plt.title('Fit to forest probs')

    # plt.subplot(195)
    plt.figure()
    plt.scatter(values,regression_probabilities1,s=s,c=colors)
    plt.plot(regression_line_xs,regression_line_1_ys,'-')
    plt.ylim([-0.1,1.1])
    plt.xlabel(x_label)
    plt.ylabel('Estimated applicability of ' + relation_phrases[0])
    plt.title('Fit to classes')

    # plt.subplot(196)
    # plt.figure()
    # plt.scatter(values,moving_average,s=s,c=colors)
    # plt.ylim([-0.1,1.1])
    # plt.title('Class Moving Average')

    # plt.subplot(197)
    plt.figure()
    plt.scatter(values,positives,s=s,c=colors)
    plt.ylim([-0.1,1.1])
    plt.xlabel(x_label)
    plt.ylabel('Presumed class of example: 1=%s, 0=%s' %(relation_phrases[0],
                                                         relation_phrases[0]))
    plt.title('Presumed Class')

    # plt.subplot(198)
    plt.figure()
    plt.scatter(values,true_scores,s=s,c=colors)
    plt.ylim([-0.1,1.1])
    plt.xlabel(x_label)
    plt.ylabel('Teacher Score of ' + relation_phrases[0])
    plt.title('True Score')

    # plt.subplot(199)
    plt.figure()
    plt.scatter(values,true_applicabilities,s=s,c=colors)
    plt.ylim([-0.1,1.1])
    plt.xlabel(x_label)
    plt.ylabel('Teacher Applicability of ' + relation_phrases[0])
    plt.title('True Applicability')

    # plt.suptitle(feature.name)
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()
    raw_input()
    # IPython.embed()



@automain
def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--db_url', required=True, type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-l', '--landmark_iterations', type=int, default=0)
    parser.add_argument('-r', '--relation_iterations', type=int, default=0)
    parser.add_argument('-s', '--scene_directory', type=str)
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-g', '--graphs_file', type=str)
    parser.add_argument('-c', '--cheat', action='store_true')
    parser.add_argument('-p', '--relation_phrases', type=str, nargs='+')
    args = parser.parse_args()

    session_factory = get_session_factory(args.db_url, echo=False)

    if args.scene_directory:
        scene_descs = read_scenes(args.scene_directory,
                                  normalize=True,image=True)

    # for scene, speaker, image in scene_descs:

    #   fig = plt.imshow(image)
    #   fig.axes.get_xaxis().set_visible(False)
    #   fig.axes.get_yaxis().set_visible(False)
    #   plt.show()

    if args.graphs_file:
        filename = args.graphs_file
    else:
        filename = 'testing'
        filename += ('_%s_%s.shelf' % (args.tag,
                                       time.asctime(time.localtime())
                                       .replace(' ','_').replace(':','')))

    # parallel = False

    # random_scene_training(session_factory1=session_factory,
    #                 iterations=args.relation_iterations,
    #                 per_scene=10,
    #                 save_file=filename,
    #                 just_landmark=False,
    #                 cheat=args.cheat,
    #                 train_only=True,
    #                 parallel=parallel)
    # exit()

    feature_columns = [
                    RelationObservation.trajector_distance,
                    RelationObservation.trajector_surface,
                    RelationObservation.trajector_contained,
                    RelationObservation.trajector_in_front_of,
                    RelationObservation.trajector_behind,
                    RelationObservation.trajector_left_of,
                    RelationObservation.trajector_right_of,
                    RelationObservation.positive,
                    RelationObservation.landmark_prior,
                    RelationObservation.true_relation_applicability
                    ]

    relation_phrase = args.relation_phrases[0]

    # session = session_factory()
    # q = session.query(Utterance.id).filter(
    #         Utterance.text.startswith(relation_phrase))
    # ids = zip(*q.all())[0]

    # for eyedee in ids:
    #     q = session.query(*feature_columns)
    #     q = q.filter(RelationObservation.utterance_id==eyedee)

    #     qall = zip(*q.all())
    #     feature_tuples = zip(*qall[:-3])
    #     label_tuples = qall[-3]
    #     weight_tuples = qall[-2]
    #     score_tuples = qall[-1]

    #     feature_array = np.rec.fromrecords(feature_tuples, 
    #                                names=[f.key for f in feature_columns[:-2]])
    #     label_array = np.array(label_tuples).flatten()
    #     weight_array = np.array(weight_tuples).flatten()
    #     score_array = np.array(score_tuples).flatten()

    #     from student_trees import trees
    #     tree_out = trees[relation_phrase](feature_array)

    #     weight_set = set(weight_array)
    #     for weight in weight_set:
    #         w = np.where(weight_array==weight)
    #         print label_array[w], tree_out[w], weight

    #     IPython.embed()

    #     exit()

    # label_column = RelationObservation.positive
    
    relation_phrase = args.relation_phrases[0]

    session = session_factory()
    q = session.query(*feature_columns)
    qall = zip(*q.all())
    feature_tuples = zip(*qall[:-3])
    label_tuples = qall[-3]
    weight_tuples = qall[-2]
    score_tuples = qall[-1]
    
    base_feature_array = np.rec.fromrecords(feature_tuples, 
                                   names=[f.key for f in feature_columns[:-2]])
    base_weights_array = np.array(weight_tuples).flatten()

    q = q.filter(RelationObservation.relation_phrase == relation_phrase)
    qall = zip(*q.all())
    feature_tuples = zip(*qall[:-3])
    label_tuples = qall[-3]
    weight_tuples = qall[-2]
    score_tuples = qall[-1]
    
    feature_array = np.rec.fromrecords(feature_tuples, 
                                   names=[f.key for f in feature_columns[:-2]])
    label_array = np.array(label_tuples).flatten()
    weight_array = np.array(weight_tuples).flatten()
    score_array = np.array(score_tuples).flatten()
    # print label_array[:10]
    # print weights_array[:10]
    # from bayes_tree import BayesianRegressionTree
    # t = BayesianRegressionTree(features=feature_array, 
    #                            labels=label_array, 
    #                            weights=weights_array, 
    #                            max_depth=5,
    #                            base_features=base_feature_array,
    #                            base_weights=base_weights_array)


    # b = brt.BernoulliRegressionTree.cv_init2(feature_array, label_array, weight_array)
    cost = 'gini'
    b00 = brt.BernoulliRegressionTree(feature_array, label_array, weight_array,
                                      cost=cost, max_split=0, max_cont=0)
    b10 = brt.BernoulliRegressionTree(feature_array, label_array, weight_array,
                                      cost=cost, max_split=1, max_cont=0)
    b20 = brt.BernoulliRegressionTree(feature_array, label_array, weight_array,
                                      cost=cost, max_split=2, max_cont=0)
    b30 = brt.BernoulliRegressionTree(feature_array, label_array, weight_array,
                                      cost=cost, max_split=3, max_cont=0)
    b01 = brt.BernoulliRegressionTree(feature_array, label_array, weight_array,
                                      cost=cost, max_split=0, max_cont=1)
    b11 = brt.BernoulliRegressionTree(feature_array, label_array, weight_array,
                                      cost=cost, max_split=1, max_cont=1)
    b21 = brt.BernoulliRegressionTree(feature_array, label_array, weight_array,
                                      cost=cost, max_split=2, max_cont=1)
    b31 = brt.BernoulliRegressionTree(feature_array, label_array, weight_array,
                                      cost=cost, max_split=3, max_cont=1)

    best = brt.BernoulliRegressionTree.cv_init3(feature_array,
                                                label_array,
                                                weight_array,
                                                cost=cost,
                                                max_split=3,
                                                max_cont=1)

    print best.tree.nodes
    # print best.tree.nodes[0].feature
    # Make some graphs

    colors = ['g' if p else 'r' for p in label_array]
    # altcolors = ['b' if p else 'k' for p in label_frame.values]
    s = 50*(4**np.array(weight_array))

    plt.ion()
    plt.subplot(251)
    plt.scatter(feature_array['trajector_distance'], score_array, s=s, c=colors)
    plt.ylim([-0.1,1.1])
    plt.subplot(252)
    plt.scatter(feature_array['trajector_distance'], b00.tree(feature_array),s=s,c=colors)
    plt.ylim([-0.1,1.1])
    plt.subplot(253)
    plt.scatter(feature_array['trajector_distance'], b10.tree(feature_array),s=s,c=colors)
    plt.ylim([-0.1,1.1])
    plt.subplot(254)
    plt.scatter(feature_array['trajector_distance'], b20.tree(feature_array),s=s,c=colors)
    plt.ylim([-0.1,1.1])
    plt.subplot(255)
    plt.scatter(feature_array['trajector_distance'], b30.tree(feature_array),s=s,c=colors)
    plt.ylim([-0.1,1.1])

    plt.subplot(256)
    plt.scatter(feature_array['trajector_distance'], best.tree(feature_array),s=s,c=colors)
    plt.ylim([-0.1,1.1])

    plt.subplot(257)
    plt.scatter(feature_array['trajector_distance'], b01.tree(feature_array),s=s,c=colors)
    plt.ylim([-0.1,1.1])
    plt.subplot(258)
    plt.scatter(feature_array['trajector_distance'], b11.tree(feature_array),s=s,c=colors)
    plt.ylim([-0.1,1.1])
    plt.subplot(259)
    plt.scatter(feature_array['trajector_distance'], b21.tree(feature_array),s=s,c=colors)
    plt.ylim([-0.1,1.1])
    plt.subplot(2,5,10)
    plt.scatter(feature_array['trajector_distance'], b31.tree(feature_array),s=s,c=colors)
    plt.ylim([-0.1,1.1])
    plt.show()
    IPython.embed()
    exit()

    import pandas as pd
    import statsmodels.api as sm

    feature_frame = pd.DataFrame(feature_array)
    feature_frame = sm.add_constant(feature_frame)
    feature_frame[feature_frame.columns[2:8]] = \
        feature_frame[feature_frame.columns[2:8]].astype(int)

    label_frame = pd.DataFrame(label_array)
    label_frame = label_frame.astype('int')

    weight_frame = pd.DataFrame(weight_array)



    logit = lambda features, params: 1./(1.+np.exp(-(params*features).sum(axis=1)))

    my_params = np.array([1,-20,-10,-10,0,0,0,0])


    # logit_mod = sm.Logit(label_frame, feature_frame, weights=weight_frame)
    # logit_res = logit_mod.fit()
    # logit_res.summary()

    # plt.scatter(feature_frame['trajector_distance'], score_array, s=s, c=altcolors)
    # plt.scatter(feature_frame['trajector_distance'], logit(feature_frame,my_params),s=s,c=colors)
    # plt.show()

    f = np.array(feature_frame)[:,1:]
    l = np.array(label_frame).flatten()
    w = np.array(weight_frame).flatten()
    import sklearn.linear_model as lm
    sgd = lm.SGDClassifier(loss='log', alpha=0.0001)
    clf = sgd.fit(f,l,sample_weight=w)#, coef_init=[-20,-10,-10, 0,0,0,0])
    sgd2 = lm.SGDClassifier(loss='log', alpha=0.0001)
    clf2 = sgd2.fit(f[:,:3],l,sample_weight=w)#, coef_init=[-20,-10,-10])

    from student_trees import trees

    tree_out = trees[relation_phrase](feature_array)
    # tree_labels = tree_out >= 0.5
    import sklearn.metrics as metrics

    precision, recall, thresholds = metrics.precision_recall_curve(label_array, tree_out)
    area = metrics.auc(recall, precision)
    print("Area Under Curve: %0.2f" % area)

    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC=%0.2f' % area)
    plt.legend(loc="lower left")
    plt.show()

    IPython.embed()
    exit()

    # make_some_histograms(session_factory)

    # random_scene_training(session_factory1=session_factory,
    #                     iterations=args.landmark_iterations,
    #                     per_scene=100,
    #                     save_file=filename,
    #                     just_landmark=True,
    #                     cheat=args.cheat)

    # random_scene_training(session_factory1=session_factory,
    #                 iterations=args.relation_iterations,
    #                 per_scene=100,
    #                 save_file=filename,
    #                 just_landmark=False,
    #                 cheat=args.cheat)

    # random_scene_training(session_factory1=session_factory,
    #             iterations=args.relation_iterations,
    #             per_scene=100,
    #             save_file=filename,
    #             just_landmark=False,
    #             test_only=True,
    #             cheat=args.cheat)

    # cross_validate(db_url=args.db_url,
    #              scene_descs=scene_descs, 
    #              filename=filename, 
    #              cheat=args.cheat)

    # regular_train(session_factory, 
    #             [(scene, speaker) for scene,speaker,_ in scene_descs], 
    #             args.landmark_iterations,
    #             save_file=filename, 
    #             just_landmark=True,
    #             cheat=args.cheat,
    #             parallel=parallel)

    # regular_train(session_factory, 
    #             [(scene, speaker) for scene,speaker,_ in scene_descs], 
    #             args.relation_iterations, 
    #             save_file=filename,
    #             just_landmark=False,
    #             cheat=args.cheat,
    #             parallel=parallel)

    # view_graphs(filename)

    # data, _,_, weightID = make_orange_table(session_factory=session_factory,
    #                   relation_phrases=['to the left of','to the right of'])
    # features, classes, weights = table.to_numpy('a/cw',weightID)
    # distance_graphs(session_factory, relation_phrases=args.relation_phrases)

    # residuals from logistic regression
    # step-wise
    # build a cascade of classifiers by hand
    # downweight other relations
    # em-like