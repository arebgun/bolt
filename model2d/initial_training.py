#!/usr/bin/env python
from __future__ import division

import sys
sys.path.append("..")

from itertools import izip

import argparse
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Process, Pipe
from myrandom import random
random = random.random
from nltk.metrics.distance import edit_distance
from planar import Vec2
from semantics.landmark import Landmark
from semantics.representation import PointRepresentation
from semantics.run import construct_training_scene

from location_from_sentence import get_all_sentence_posteriors
from sentence_from_location import generate_sentence, train
from utils import logger, entropy_of_probs
import utils

import shelve
from time import time

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

def initial_train(num_iterations=1, window=20, num_processors=1, consistent=False,
                  num_samples=10, num_per_scene=50):
    # plt.ion()
    printing=False

    def loop(num_iterations):
        # min_dists = []
        # golden_log_probs = []
        # golden_entropies = []
        # golden_ranks = []
        # golden_ranks = []

        for iteration in range(num_iterations):
            if (iteration % num_per_scene) == 0:
                if iteration != 0: sys.stdout.write('\bDone.\n')
                sys.stdout.write('Generating sentences for scene %d:  ' % (int(iteration/num_per_scene) + 1))
                scene, speaker = construct_training_scene(random=True)
                utils.scene.set_scene(scene, speaker)
                table = scene.landmarks['table'].representation.rect

                # scene_bb = scene.get_bounding_box()
                # scene_bb = scene_bb.inflate( Vec2(scene_bb.width*0.5,scene_bb.height*0.5) )

                # step = 0.1
                # all_heatmaps_tupless, xs, ys = speaker.generate_all_heatmaps(scene, step=step)
                # all_heatmaps_tuples = all_heatmaps_tupless[0]
                # lmks, rels, heatmapss = zip(*all_heatmaps_tuples)

                # meanings = zip(lmks,rels)
                # landmarks = list(set(lmks))

            logger(('Iteration %d' % iteration),'okblue')

            rand_p = Vec2(random()*table.width+table.min_point.x, random()*table.height+table.min_point.y)
            trajector = Landmark( 'point', PointRepresentation(rand_p), None, Landmark.POINT)
            training_sentence, sampled_relation, sampled_landmark = speaker.describe(trajector, scene, False, 1)

            if num_samples:
                for i in range(num_samples):
                    landmark, relation, _ = speaker.sample_meaning(trajector, scene, 1)
                    train((landmark,relation), training_sentence, update=1, printing=printing)
            else:
                for (landmark,relation),prob in speaker.all_meaning_probs( trajector, scene, 1 ):
                    train((landmark,relation), training_sentence, update=prob, printing=printing)

        #     def probs_metric():
        #         meaning, sentence = generate_sentence(loc=rand_p,
        #                                               consistent=consistent,
        #                                               scene=scene,
        #                                               speaker=speaker,
        #                                               usebest=True,
        #                                               printing=printing)

        #         sampled_landmark, sampled_relation = meaning.args[0], meaning.args[3]
        #         print meaning.args[0], meaning.args[3], len(sentence)

        #         if sentence == "":
        #             prob = 0
        #             entropy = 0
        #             rank = len(meanings)-1
        #         else:
        #             logger( 'Generated sentence: %s' % sentence)

        #             try:
        #                 golden_posteriors = get_all_sentence_posteriors(sentence, meanings, golden=True, printing=printing)
        #                 epsilon = 1e-15
        #                 ps = np.array([golden_posteriors[(lmk,rel)] for lmk, rel in meanings])
        #                 temp = None
        #                 for i,p in enumerate(ps):
        #                     lmk,rel = meanings[i]
        #                     # logger( '%f, %s' % (p, m2s(lmk,rel)))
        #                     head_on = speaker.get_head_on_viewpoint(lmk)
        #                     ps[i] *= speaker.get_landmark_probability(lmk, landmarks, PointRepresentation(rand_p))[0]
        #                     ps[i] *= speaker.get_probabilities_points(np.array([rand_p]), rel, head_on, lmk)
        #                     if lmk == meaning.args[0] and rel == meaning.args[3]:
        #                         temp = i

        #                 ps += epsilon
        #                 ps = ps/ps.sum()
        #                 prob = ps[temp]
        #                 rank = sorted(ps, reverse=True).index(prob)
        #                 entropy = entropy_of_probs(ps)
        #             except Exception as pe:
        #                 logger('Failed to parse sentence "%s"' % sentence, 'warning')
        #                 logger(str(pe), 'warning')

        #                 prob = 0
        #                 rank = len(meanings)-1
        #                 entropy = 0

        #         head_on = speaker.get_head_on_viewpoint(sampled_landmark)
        #         all_descs = speaker.get_all_meaning_descriptions(trajector, scene, sampled_landmark, sampled_relation, head_on, 1)
        #         distances = []

        #         for desc in all_descs:
        #             distances.append([edit_distance( sentence, desc ), desc])

        #         distances.sort()

        #         return prob,entropy,rank,distances[0][0]


        #     prob, entropy, rank, ed = probs_metric()

        #     golden_log_probs.append( prob )
        #     golden_entropies.append( entropy )
        #     golden_ranks.append( rank )
        #     min_dists.append( ed )

        # return zip(golden_log_probs, golden_entropies, golden_ranks, min_dists)

    num_each = int(num_iterations/num_processors)
    num_iterationss = [num_each]*num_processors
    # num_iterationss[-1] += num_iterations-num_each*num_processors
    logger( num_iterationss )
    # lists = parmap(loop,num_iterationss)
    parmap(loop,num_iterationss)
    logger( '%s, %s' % (num_processors,num_each) )
    # print lists
    # print len(lists), len(lists[0])
    # result = []
    # for i in range(num_each):
    #     logger( i )
    #     for j in range(num_processors):
    #         logger( '  %i %i %i' % (j,len(lists),len(lists[j])) )
    #         result.append( lists[j][i] )

    # golden_log_probs,golden_entropies,golden_ranks,min_dists = zip(*result)

    # def running_avg(arr):
    #     return [np.mean(arr[i:i+window]) for i in range(len(arr)-window)]

    # avg_golden_log_probs = running_avg(golden_log_probs)
    # avg_golden_entropies = running_avg(golden_entropies)
    # avg_golden_ranks = running_avg(golden_ranks)
    # avg_min = running_avg(min_dists)

    # filename = 'initial_training_%d_samples_%d_iters_%d_per_scene_%f.shelf' % (num_samples, num_iterations, num_per_scene, time())
    # f = shelve.open(filename)
    # f['golden_log_probs'] = golden_log_probs
    # f['golden_entropies'] = golden_entropies
    # f['golden_ranks'] = golden_ranks
    # f['min_dists'] = min_dists
    # f['avg_golden_log_probs'] = avg_golden_log_probs
    # f['avg_golden_entropies'] = avg_golden_entropies
    # f['avg_golden_ranks'] = avg_golden_ranks
    # f['avg_min'] = avg_min
    # f.close()

    # plt.plot(avg_min, 'o-', color='RoyalBlue')
    # plt.ylabel('Edit Distance')
    # plt.title('Initial Training')
    # plt.show()
    # plt.draw()

    # plt.figure()
    # plt.suptitle('Initial Training')
    # plt.subplot(211)
    # plt.plot(golden_log_probs, 'o-', color='RoyalBlue')
    # plt.plot(avg_golden_log_probs, 'x-', color='Orange')
    # plt.ylabel('Golden Probability')

    # plt.subplot(212)
    # plt.plot(golden_ranks, 'o-', color='RoyalBlue')
    # plt.plot(avg_golden_ranks, 'x-', color='Orange')
    # plt.ylabel('Golden Rank')
    # plt.ioff()
    # plt.show()
    # plt.draw()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-iterations', type=int, default=1000)
    parser.add_argument('-i', '--num-per-scene', type=int, default=50)
    parser.add_argument('-p', '--num-processors', type=int, default=1)
    parser.add_argument('-w', '--window-size', type=int, default=20)
    parser.add_argument('-s', '--num-samples', type=int, default=10)
    parser.add_argument('--consistent', action='store_true')
    args = parser.parse_args()

    initial_train(args.num_iterations, window=args.window_size,
        num_processors=args.num_processors, consistent=args.consistent,
        num_samples=args.num_samples, num_per_scene=args.num_per_scene)



