#!/usr/bin/env python
from __future__ import division

# from random import random
import sys
import traceback
sys.path.append("..")
from myrandom import random
choice = random.choice
random = random.random

from sentence_from_location import (
    generate_sentence,
    accept_correction,
    accept_object_correction,
    train,
    Point
)

from location_from_sentence import get_tree_probs
from parse import get_modparse, ParseError
from nltk.tree import ParentedTree

from semantics.run import construct_training_scene
from semantics.landmark import Landmark
from semantics.representation import PointRepresentation, LineRepresentation, RectangleRepresentation, GroupLineRepresentation
from nltk.metrics.distance import edit_distance
from planar import Vec2
from utils import logger, m2s, entropy_of_probs, printcolors
import numpy as np
from matplotlib import pyplot as plt
from copy import copy
from datetime import datetime

from location_from_sentence import get_all_sentence_posteriors
from multiprocessing import Process, Pipe
from itertools import izip, product
from models import CProduction, CWord

from utils import categorical_sample

from semantics.language_generator import describe
# import IPython
# IPython.embed()

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

def autocorrect(scene, speaker, num_iterations=1, scale=1000, num_processors=7, num_samples=5, 
                golden_metric=True, mass_metric=True, student_metric=True, choosing_metric=True):
    plt.ion()

    printing=False

    scene_bb = scene.get_bounding_box()
    scene_bb = scene_bb.inflate( Vec2(scene_bb.width*0.5,scene_bb.height*0.5) )
    table = scene.landmarks['table'].representation.get_geometry()

    step = 0.04
    loi = [lmk for lmk in scene.landmarks.values() if lmk.name != 'table']
    all_heatmaps_tupless, xs, ys = speaker.generate_all_heatmaps(scene, step=step, loi=loi)

    loi_infos = []
    all_meanings = set()
    for obj_lmk,all_heatmaps_tuples in zip(loi, all_heatmaps_tupless):
        
        lmks, rels, heatmapss = zip(*all_heatmaps_tuples)
        meanings = zip(lmks,rels)
        # print meanings
        all_meanings.update(meanings)
        loi_infos.append( (obj_lmk, meanings, heatmapss) )

    all_heatmaps_tupless, xs, ys = speaker.generate_all_heatmaps(scene, step=step)
    all_heatmaps_tuples = all_heatmaps_tupless[0]
    x = np.array( [list(xs-step*0.5)]*len(ys) )
    y = np.array( [list(ys-step*0.5)]*len(xs) ).T
    lmks, rels, heatmapss = zip(*all_heatmaps_tuples)
    graphmax1 = graphmax2 = 0
    meanings = zip(lmks,rels)
    landmarks = list(set(lmks))
    relations = list(set(rels))

    epsilon = 0.0001
    def heatmaps_for_sentence(sentence, all_meanings, loi_infos, xs, ys, scene, speaker, step=0.02):
        printing=False
        scene_bb = scene.get_bounding_box()
        scene_bb = scene_bb.inflate( Vec2(scene_bb.width*0.5,scene_bb.height*0.5) )
        x = np.array( [list(xs-step*0.5)]*len(ys) )
        y = np.array( [list(ys-step*0.5)]*len(xs) ).T

        posteriors = get_all_sentence_posteriors(sentence, all_meanings, printing=printing)

        combined_heatmaps = []
        for obj_lmk, ms, heatmapss in loi_infos:

            big_heatmap1 = None
            for m,(h1,h2) in zip(ms, heatmapss):
                lmk,rel = m
                p = posteriors[rel]*posteriors[lmk]
                if big_heatmap1 is None:
                    big_heatmap1 = p*h1
                else:
                    big_heatmap1 += p*h1

            combined_heatmaps.append(big_heatmap1)

        return combined_heatmaps

    object_meaning_applicabilities = {}
    for obj_lmk, ms, heatmapss in loi_infos:
        for m,(h1,_) in zip(ms, heatmapss):
            ps = [p for (x,y),p in zip(list(product(xs,ys)),h1) if obj_lmk.representation.contains_point( Vec2(x,y) )]
            if m not in object_meaning_applicabilities:
                object_meaning_applicabilities[m] = {}
            object_meaning_applicabilities[m][obj_lmk] = sum(ps)/len(ps)

    k = len(loi)
    for meaning_dict in object_meaning_applicabilities.values():
        total = sum( meaning_dict.values() )
        if total != 0:
            for obj_lmk in meaning_dict.keys():
                meaning_dict[obj_lmk] = meaning_dict[obj_lmk]/total - 1.0/k
            total = sum( [value for value in meaning_dict.values() if value > 0] )
            for obj_lmk in meaning_dict.keys():
                meaning_dict[obj_lmk] = (2 if meaning_dict[obj_lmk] > 0 else 1)*meaning_dict[obj_lmk] - total

    sorted_meaning_lists = {}

    for m in object_meaning_applicabilities.keys():
        for obj_lmk in object_meaning_applicabilities[m].keys():
            if obj_lmk not in sorted_meaning_lists:
                sorted_meaning_lists[obj_lmk] = []
            sorted_meaning_lists[obj_lmk].append( (object_meaning_applicabilities[m][obj_lmk], m) )
    for obj_lmk in sorted_meaning_lists.keys():
        sorted_meaning_lists[obj_lmk].sort(reverse=True)

    def loop(num_iterations):
        min_dists = []
        lmk_priors = []
        rel_priors = []
        lmk_posts = []
        rel_posts = []
        golden_log_probs = []
        golden_entropies = []
        golden_ranks = []
        rel_types = []

        total_mass = []

        student_probs = []
        student_entropies = []
        student_ranks = []
        student_rel_types = []

        object_answers = []
        object_distributions = []

        epsilon = 1e-15
        for iteration in range(num_iterations):
            logger(('Iteration %d' % iteration),'okblue')

            # Teacher describe 
            trajector = choice(loi)
            # sentence, sampled_relation, sampled_landmark = speaker.describe(trajector, scene, max_level=1)
            logger( 'Teacher chooses: %s' % trajector )
            # Choose from meanings
            probs, sorted_meanings = zip(*sorted_meaning_lists[trajector])
            (sampled_landmark, sampled_relation) = categorical_sample( sorted_meanings, probs )[0]
            logger( 'Teacher tries to say: %s, %s' % (sampled_landmark,sampled_relation) )

            # Generate sentence
            # _, sentence = generate_sentence(None, False, scene, speaker, meaning=(sampled_landmark, sampled_relation), golden=True, printing=printing)

            sentence = describe( speaker.get_head_on_viewpoint(sampled_landmark), trajector, sampled_landmark, sampled_relation )
            logger( 'Teacher says: %s' % sentence)

            lmk_probs = []
            try:
                combined_heatmaps = heatmaps_for_sentence(sentence, all_meanings, loi_infos, xs, ys, scene, speaker, step=step)
                
                for combined_heatmap,obj_lmk in zip(combined_heatmaps, loi):
                    ps = [p for (x,y),p in zip(list(product(xs,ys)),combined_heatmap) if obj_lmk.representation.contains_point( Vec2(x,y) )]
                    # print ps, xs.shape, ys.shape, combined_heatmap.shape
                    lmk_probs.append( (sum(ps)/len(ps), obj_lmk) )
                  
                lmk_probs = sorted(lmk_probs, reverse=True)
                top_p, top_lmk = lmk_probs[0]
                lprobs, lmkss = zip(*lmk_probs)
                
                logger( sorted(zip(np.array(lprobs)/sum(lprobs), [(l.name, l.color, l.object_class) for l in lmkss]), reverse=True) )
                logger( 'I bet %f you are talking about a %s %s %s' % (top_p/sum(lprobs), top_lmk.name, top_lmk.color, top_lmk.object_class) )
                # objects.append(top_lmk)
            except Exception as e:
                logger( 'Unable to get object from sentence. %s' % e, 'fail' )
                print traceback.format_exc()
                exit()

            # Present top_lmk to teacher
            if top_lmk == trajector:
                # Give morphine
                pass
            else:
                howmany=5
                for update, meaning in sorted_meaning_lists[trajector][:howmany]:
                    accept_object_correction( meaning, sentence, update*scale, printing=printing)
                for update, meaning in sorted_meaning_lists[trajector][-howmany:]:
                    accept_object_correction( meaning, sentence, update*scale, printing=printing)

 
            def probs_metric(inverse=False):
                rand_p = Vec2(random()*table.width+table.min_point.x, random()*table.height+table.min_point.y)
                bestmeaning, bestsentence = generate_sentence(rand_p, False, scene, speaker, usebest=True, golden=inverse, printing=printing)
                sampled_landmark, sampled_relation = bestmeaning.args[0], bestmeaning.args[3]
                try:
                    golden_posteriors = get_all_sentence_posteriors(bestsentence, meanings, golden=(not inverse), printing=printing)

                    # lmk_prior = speaker.get_landmark_probability(sampled_landmark, landmarks, PointRepresentation(rand_p))[0]
                    all_lmk_probs = speaker.all_landmark_probs(landmarks, Landmark(None, PointRepresentation(rand_p), None))
                    all_lmk_probs = dict(zip(landmarks, all_lmk_probs))

                    lmk_prior = all_lmk_probs[sampled_landmark]
                    head_on = speaker.get_head_on_viewpoint(sampled_landmark)
                    rel_prior = speaker.get_probabilities_points( np.array([rand_p]), sampled_relation, head_on, sampled_landmark)
                    lmk_post = golden_posteriors[sampled_landmark]
                    rel_post = golden_posteriors[sampled_relation]

                    ps = np.array([golden_posteriors[lmk]*golden_posteriors[rel] for lmk, rel in meanings])
                    rank = None
                    for i,p in enumerate(ps):
                        lmk,rel = meanings[i]
                        # logger( '%f, %s' % (p, m2s(lmk,rel)))
                        head_on = speaker.get_head_on_viewpoint(lmk)
                        # ps[i] *= speaker.get_landmark_probability(lmk, landmarks, PointRepresentation(rand_p))[0]
                        ps[i] *= all_lmk_probs[lmk]
                        ps[i] *= speaker.get_probabilities_points( np.array([rand_p]), rel, head_on, lmk)
                        if lmk == sampled_landmark and rel == sampled_relation:
                            idx = i

                    ps += epsilon
                    ps = ps/ps.sum()
                    prob = ps[idx]
                    rank = sorted(ps, reverse=True).index(prob)
                    entropy = entropy_of_probs(ps)
                except ParseError as e:
                    logger( e )
                    prob = 0
                    rank = len(meanings)-1
                    entropy = 0

                head_on = speaker.get_head_on_viewpoint(sampled_landmark)
                all_descs = speaker.get_all_meaning_descriptions(trajector, scene, sampled_landmark, sampled_relation, head_on, 1)
                distances = []
                for desc in all_descs:
                    distances.append([edit_distance( bestsentence, desc ), desc])
                distances.sort()
                return lmk_prior,rel_prior,lmk_post,rel_post,\
                       prob,entropy,rank,distances[0][0],type(sampled_relation)

            def db_mass():
                total = CProduction.get_production_sum(None)
                total += CWord.get_word_sum(None)
                return total

            def choosing_object_metric():
                trajector = choice(loi)

                sentence, sampled_relation, sampled_landmark = speaker.describe(trajector, scene, max_level=1)

                lmk_probs = []
                try:
                    combined_heatmaps = heatmaps_for_sentence(sentence, all_meanings, loi_infos, xs, ys, scene, speaker, step=step)
                    
                    for combined_heatmap,obj_lmk in zip(combined_heatmaps, loi):
                        ps = [p for (x,y),p in zip(list(product(xs,ys)),combined_heatmap) if obj_lmk.representation.contains_point( Vec2(x,y) )]
                        # print ps, xs.shape, ys.shape, combined_heatmap.shape
                        lmk_probs.append( (sum(ps)/len(ps), obj_lmk) )
                      
                    lmk_probs = sorted(lmk_probs, reverse=True)
                    top_p, top_lmk = lmk_probs[0]
                    lprobs, lmkss = zip(*lmk_probs)
                    
                    logger( sorted(zip(np.array(lprobs)/sum(lprobs), [(l.name, l.color, l.object_class) for l in lmkss]), reverse=True) )
                    logger( 'I bet %f you are talking about a %s %s %s' % (top_p/sum(lprobs), top_lmk.name, top_lmk.color, top_lmk.object_class) )
                    # objects.append(top_lmk)
                except Exception as e:
                    logger( 'Unable to get object from sentence. %s' % e, 'fail' )
                    print traceback.format_exc()
                    exit()
                return loi.index(trajector), [ (lprob, loi.index(lmk)) for lprob,lmk in lmk_probs ]

            if golden_metric:
                lmk_prior,rel_prior,lmk_post,rel_post,prob,entropy,rank,ed,rel_type = probs_metric()
            else:
                lmk_prior,rel_prior,lmk_post,rel_post,prob,entropy,rank,ed,rel_type = \
                None, None, None, None, None, None, None, None, None

            lmk_priors.append( lmk_prior )
            rel_priors.append( rel_prior )
            lmk_posts.append( lmk_post )
            rel_posts.append( rel_post )
            golden_log_probs.append( prob )
            golden_entropies.append( entropy )
            golden_ranks.append( rank )
            min_dists.append( ed )
            rel_types.append( rel_type )

            if mass_metric:
                total_mass.append( db_mass() )
            else:
                total_mass.append( None )

            if student_metric:
                _,_,_,_,student_prob,student_entropy,student_rank,_,student_rel_type = probs_metric(inverse=True)
            else:
                _,_,_,_,student_prob,student_entropy,student_rank,_,student_rel_type = \
                None, None, None, None, None, None, None, None, None

            student_probs.append( student_prob )
            student_entropies.append( student_entropy )
            student_ranks.append( student_rank )
            student_rel_types.append( student_rel_type )

            if choosing_metric:
                answer, distribution = choosing_object_metric()
            else:
                answer, distribution = None, None
            object_answers.append( answer )
            object_distributions.append( distribution )

        return zip(lmk_priors, rel_priors, lmk_posts, rel_posts,
                   golden_log_probs, golden_entropies, golden_ranks, 
                   min_dists, rel_types, total_mass, student_probs, 
                   student_entropies, student_ranks, student_rel_types,
                   object_answers, object_distributions)

    filename = 'objcorr'
    filename += ('_p%i_n%i_u%i.shelf' % (num_processors,num_iterations,scale))
    import shelve
    f = shelve.open(filename)
    f['lmk_priors']           = []
    f['rel_priors']           = []
    f['lmk_posts']            = []
    f['rel_posts']            = []
    f['golden_log_probs']     = []
    f['golden_entropies']     = []
    f['golden_ranks']         = []
    f['min_dists']            = []
    f['rel_types']            = []
    f['total_mass']           = []
    f['student_probs']        = []
    f['student_entropies']    = []
    f['student_ranks']        = []
    f['student_rel_types']    = []
    f['object_answers']       = []
    f['object_distributions'] = []
    # f['initial_training']     = initial_training
    # f['cheating']             = cheating
    # f['explicit_pointing']    = explicit_pointing
    # f['ambiguous_pointing']   = ambiguous_pointing
    f.close()

    chunk_size = 10
    num_each = int(num_iterations/num_processors)
    n = int(num_each / chunk_size)
    extra = num_each % chunk_size
    logger( "num_each: %i, chunk_size: %i, n: %i, extra: %i" % (num_each, chunk_size, n, extra) )

    for i in range(n):
        lists = parmap(loop,[chunk_size]*num_processors)
        # lists = map(loop,[chunk_size]*num_processors)

        result = []
        for i in range(chunk_size):
	        for j in range(num_processors):
	            result.append( lists[j][i] )
        lmk_priors, rel_priors, lmk_posts, rel_posts, \
            golden_log_probs, golden_entropies, golden_ranks, \
            min_dists, rel_types, total_mass, student_probs, student_entropies, \
            student_ranks, student_rel_types, object_answers, object_distributions = zip(*result)
        f = shelve.open(filename)
        f['lmk_priors']           += lmk_priors
        f['rel_priors']           += rel_priors
        f['lmk_posts']            += lmk_posts
        f['rel_posts']            += rel_posts
        f['golden_log_probs']     += golden_log_probs
        f['golden_entropies']     += golden_entropies
        f['golden_ranks']         += golden_ranks
        f['min_dists']            += min_dists
        f['rel_types']            += rel_types
        f['total_mass']           += total_mass
        f['student_probs']        += student_probs
        f['student_entropies']    += student_entropies
        f['student_ranks']        += student_ranks
        f['student_rel_types']    += student_rel_types
        f['object_answers']       += object_answers
        f['object_distributions'] += object_distributions
        f.close()
        
    if extra:
        lists = parmap(loop,[extra]*num_processors)
        # lists = map(loop,[extra]*num_processors)
        result = []
        for i in range(extra):
	        for j in range(num_processors):
	            result.append( lists[j][i] )
        lmk_priors, rel_priors, lmk_posts, rel_posts, \
            golden_log_probs, golden_entropies, golden_ranks, \
            min_dists, rel_types, total_mass, student_probs, student_entropies, \
            student_ranks, student_rel_types, object_answers, object_distributions = zip(*result)
        f = shelve.open(filename)
        f['lmk_priors']           += lmk_priors
        f['rel_priors']           += rel_priors
        f['lmk_posts']            += lmk_posts
        f['rel_posts']            += rel_posts
        f['golden_log_probs']     += golden_log_probs
        f['golden_entropies']     += golden_entropies
        f['golden_ranks']         += golden_ranks
        f['min_dists']            += min_dists
        f['rel_types']            += rel_types
        f['total_mass']           += total_mass
        f['student_probs']        += student_probs
        f['student_entropies']    += student_entropies
        f['student_ranks']        += student_ranks
        f['student_rel_types']    += student_rel_types
        f['object_answers']       += object_answers
        f['object_distributions'] += object_distributions
        f.close()

    exit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_iterations', type=int, default=1)
    parser.add_argument('-u', '--update_scale', type=int, default=1000)
    parser.add_argument('-p', '--num_processors', type=int, default=7)
    parser.add_argument('-s', '--num_samples', action='store_true')
    args = parser.parse_args()

    scene, speaker = construct_training_scene()

    autocorrect(scene, speaker, args.num_iterations, # window=args.window_size, 
        scale=args.update_scale, num_processors=args.num_processors, num_samples=args.num_samples,)


