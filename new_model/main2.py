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



    test_teacher(scene_descs)


    # residuals from logistic regression
    # step-wise
    # build a cascade of classifiers by hand
    # downweight other relations
    # em-like