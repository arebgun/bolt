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
from semantics.relation import DistanceRelation
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
from math import ceil

from utils import categorical_sample
import utils
import time
import shelve

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

def autocorrect(scene_descs, test_scene_descs, tag='', chunksize=5, scale=1000, num_processors=7, num_samples=5, step=0.04):
    # plt.ion()

    printing=False

    def loop(data):


        scene = data['scene']
        speaker = data['speaker']
        utils.scene.set_scene(scene,speaker)
        num_iterations = len(data['loc_descs'])

        all_meanings = data['all_meanings']
        loi_infos = data['loi_infos']
        landmarks = data['landmarks']
        sorted_meaning_lists = data['sorted_meaning_lists']
        learn_objects = data['learn_objects']

        def heatmaps_for_sentences(sentences, all_meanings, loi_infos, xs, ys, scene, speaker, step=0.02):
            printing=False
            x = np.array( [list(xs-step*0.5)]*len(ys) )
            y = np.array( [list(ys-step*0.5)]*len(xs) ).T
            scene_bb = scene.get_bounding_box()
            scene_bb = scene_bb.inflate( Vec2(scene_bb.width*0.5,scene_bb.height*0.5) )

            combined_heatmaps = []
            for obj_lmk, ms, heatmapss in loi_infos:

                combined_heatmap = None
                for sentence in sentences:
                    posteriors = get_all_sentence_posteriors(sentence, all_meanings, printing=printing)

                    big_heatmap1 = None
                    for m,(h1,h2) in zip(ms, heatmapss):

                        lmk,rel = m
                        p = posteriors[rel]*posteriors[lmk]
                        if big_heatmap1 is None:
                            big_heatmap1 = p*h1
                        else:
                            big_heatmap1 += p*h1

                    if combined_heatmap is None:
                        combined_heatmap = big_heatmap1
                    else:
                        combined_heatmap *= big_heatmap1

                combined_heatmaps.append(combined_heatmap)

            return combined_heatmaps

        object_answers = []
        object_distributions = []
        object_sentences =[]

        epsilon = 1e-15

        for iteration in range(num_iterations):
            logger(('Iteration %d comprehension' % iteration),'okblue')

            trajector = data['lmks'][iteration]
            logger( 'Teacher chooses: %s' % trajector )
            sentences = data['loc_descs'][iteration]
            probs, sorted_meanings = zip(*sorted_meaning_lists[trajector][:30])
            probs = np.array(probs)# - min(probs)
            probs /= probs.sum()
            if sentences is None:
                (sampled_landmark, sampled_relation) = categorical_sample( sorted_meanings, probs )[0]
                logger( 'Teacher tries to say: %s' % m2s(sampled_landmark,sampled_relation) )
                head_on = speaker.get_head_on_viewpoint(sampled_landmark)

                sentences = [describe( head_on, trajector, sampled_landmark, sampled_relation )]

            object_sentences.append( ' '.join(sentences) )
            logger( 'Teacher says: %s' % ' '.join(sentences))
            for i,(p,sm) in enumerate(zip(probs[:15],sorted_meanings[:15])):
                lm,re = sm
                logger( '%i: %f %s' % (i,p,m2s(*sm)) )

            lmk_probs = []

            try:
                combined_heatmaps = heatmaps_for_sentences(sentences, all_meanings, loi_infos, xs, ys, scene, speaker, step=step)

                for combined_heatmap,obj_lmk in zip(combined_heatmaps, loi):

                    ps = [p for (x,y),p in zip(list(product(xs,ys)),combined_heatmap) if obj_lmk.representation.contains_point( Vec2(x,y) )]
                    # print ps, xs.shape, ys.shape, combined_heatmap.shape
                    lmk_probs.append( (sum(ps)/len(ps), obj_lmk) )

                lmk_probs = sorted(lmk_probs, reverse=True)
                top_p, top_lmk = lmk_probs[0]
                lprobs, lmkss = zip(*lmk_probs)

                answer, distribution = loi.index(trajector), [ (lprob, loi.index(lmk)) for lprob,lmk in lmk_probs ]
                logger( sorted(zip(np.array(lprobs)/sum(lprobs), [(l.name, l.color, l.object_class) for l in lmkss]), reverse=True) )
                logger( 'I bet %f you are talking about a %s %s %s' % (top_p/sum(lprobs), top_lmk.name, top_lmk.color, top_lmk.object_class) )
                # objects.append(top_lmk)
            except Exception as e:
                logger( 'Unable to get object from sentence. %s' % e, 'fail' )
                answer = None
                top_lmk = None
                distribution = [(0,False)]

            object_answers.append( answer )
            object_distributions.append( distribution )

            # Present top_lmk to teacher
            if top_lmk == trajector or not learn_objects:
                # Give morphine
                pass
            else:
                updates, _ = zip(*sorted_meaning_lists[trajector][:30])
                howmany=5
                for sentence in sentences:
                    for _ in range(howmany):
                        meaning = categorical_sample( sorted_meanings, probs )[0]
                        update = updates[ sorted_meanings.index(meaning) ]
                        try:
                            accept_object_correction( meaning, sentence, update*scale, printing=printing)
                        except:
                            pass
                    for update, meaning in sorted_meaning_lists[trajector][-howmany:]:
                        try:
                            accept_object_correction( meaning, sentence, update*scale, printing=printing)
                        except:
                            pass

        return zip(object_answers, object_distributions, object_sentences)

    filename = 'testing'
    filename += ('_u%i_%s_%s.shelf' % (scale,tag,time.asctime(time.localtime()).replace(' ','_').replace(':','')))
    f = shelve.open(filename)
    f['object_answers']       = []
    f['object_distributions'] = []
    f['object_sentences']     = []

    f['test_object_answers']       = []
    f['test_object_distributions'] = []
    f['test_object_sentences']     = []
    f.close()

    def interleave(*args):
        for idx in range(0, max(len(arg) for arg in args)):
            for arg in args:
                try:
                    yield arg[idx]
                except IndexError:
                    continue

    def chunks(l, n):
        return [l[i:i+n] for i in range(0, len(l), n)]

    num_scenes = len(scene_descs)
    processors_per_scene = int(num_processors/num_scenes)
    # new_scene_descs = scene_descs
    new_scene_descs = []

    for scene_desc in scene_descs:

        scene = scene_desc['scene']
        speaker = scene_desc['speaker']

        utils.scene.set_scene(scene,speaker)
        scene_bb = scene.get_bounding_box()
        scene_bb = scene_bb.inflate( Vec2(scene_bb.width*0.5,scene_bb.height*0.5) )
        table = scene.landmarks['table'].representation.get_geometry()
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

        all_heatmaps_tuples = speaker.generate_all_heatmaps(scene, step=step)[0][0]
        landmarks = list(set(zip(*all_heatmaps_tuples)[0]))

        object_meaning_applicabilities = {}
        for obj_lmk, ms, heatmapss in loi_infos:
            for m,(h1,h2) in zip(ms, heatmapss):
                ps = [p for (x,y),p in zip(list(product(xs,ys)),h1) if obj_lmk.representation.contains_point( Vec2(x,y) )]
                if m not in object_meaning_applicabilities:
                    object_meaning_applicabilities[m] = {}
                object_meaning_applicabilities[m][obj_lmk] = sum(ps)/len(ps)

        k = len(loi)
        for meaning_dict in object_meaning_applicabilities.values():
            total = sum( meaning_dict.values() )
            if total != 0:
                for obj_lmk in meaning_dict.keys():
                    meaning_dict[obj_lmk] *= meaning_dict[obj_lmk]/total# - 1.0/k
        #         total = sum( [value for value in meaning_dict.values() if value > 0] )
        #         for obj_lmk in meaning_dict.keys():
        #             meaning_dict[obj_lmk] = (2 if meaning_dict[obj_lmk] > 0 else 1)*meaning_dict[obj_lmk] - total

        sorted_meaning_lists = {}

        for m in object_meaning_applicabilities.keys():
            for obj_lmk in object_meaning_applicabilities[m].keys():
                if obj_lmk not in sorted_meaning_lists:
                    sorted_meaning_lists[obj_lmk] = []
                sorted_meaning_lists[obj_lmk].append( (object_meaning_applicabilities[m][obj_lmk], m) )
        for obj_lmk in sorted_meaning_lists.keys():
            sorted_meaning_lists[obj_lmk].sort(reverse=True)

        together = zip(scene_desc['lmks'],scene_desc['loc_descs'],scene_desc['ids'])
        n = int(ceil(len(together)/float(processors_per_scene)))
        for chunk in chunks(together,n):
            lmks, loc_descs, ids = zip(*chunk)
            new_scene_descs.append( {'scene':scene_desc['scene'],
                                     'speaker':scene_desc['speaker'],
                                     'lmks':lmks,
                                     'loc_descs':loc_descs,
                                     'ids':ids,
                                     'all_meanings':all_meanings,
                                     'loi_infos':loi_infos,
                                     'landmarks':landmarks,
                                     'sorted_meaning_lists':sorted_meaning_lists,
                                     'learn_objects':True})

    # chunksize = 5
    proc_batches = []
    for scene in new_scene_descs:
        proc_batch = []
        for chunk in chunks(zip(scene['lmks'],scene['loc_descs'],scene['ids']),chunksize):
            lmks, loc_descs, ids = zip(*chunk)
            proc_batch.append({
                 'scene':scene_desc['scene'],
                 'speaker':scene_desc['speaker'],
                 'all_meanings':scene['all_meanings'],
                 'loi_infos':scene['loi_infos'],
                 'landmarks':scene['landmarks'],
                 'sorted_meaning_lists':scene['sorted_meaning_lists'],
                 'learn_objects':scene['learn_objects'],
                 'lmks':lmks,
                 'loc_descs':loc_descs,
                 'ids':ids})
        proc_batches.append(proc_batch)
    batches = map(None,*proc_batches)
    batches = map(lambda x: filter(None,x), batches)


    for scene in test_scene_descs:
        scene['learn_objects']=False

    print len(batches)
    for batch in batches:
        print ' ',len(batch)

    for batch in batches:
        lists = parmap(loop,batch)
        # lists = map(loop,new_scene_descs)
        result = list(interleave(*lists))
        object_answers, object_distributions, object_sentences = zip(*result)

        test_lists = parmap(loop,test_scene_descs)
        test_result = list(interleave(*lists))
        test_object_answers, test_object_distributions, test_object_sentences = zip(*test_result)

        f = shelve.open(filename)
        f['object_answers']       += object_answers
        f['object_distributions'] += object_distributions
        f['object_sentences']     += object_sentences

        f['test_object_answers']       .append(test_object_answers)
        f['test_object_distributions'] .append(test_object_distributions)
        f['test_object_sentences']     .append(test_object_sentences)

        f.close()



    logger("Exiting")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_iterations', type=int, default=1)
    parser.add_argument('-u', '--update_scale', type=int, default=1000)
    parser.add_argument('-p', '--num_processors', type=int, default=7)
    parser.add_argument('-s', '--num_samples', action='store_true')
    args = parser.parse_args()

    # scene, speaker = construct_training_scene()

    autocorrect(args.num_iterations, # window=args.window_size,
        scale=args.update_scale, num_processors=args.num_processors, num_samples=args.num_samples,)


