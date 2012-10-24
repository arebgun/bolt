#!/usr/bin/env python
from __future__ import division

# from random import random
import sys
sys.path.append("..")
from myrandom import random
random = random.random

from sentence_from_location import (
    generate_sentence,
    accept_correction,
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
from itertools import izip

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

def autocorrect(scene, speaker, num_iterations=1, window=10, scale=1000, consistent=False,
                cheating=False, explicit_pointing=False, ambiguous_pointing=False):
    plt.ion()

    assert cheating + explicit_pointing + ambiguous_pointing == 1, \
        'Must choose Cheating, Explicit or Ambiguous'

    printing=False

    scene_bb = scene.get_bounding_box()
    scene_bb = scene_bb.inflate( Vec2(scene_bb.width*0.5,scene_bb.height*0.5) )
    table = scene.landmarks['table'].representation.get_geometry()

    step = 0.1
    all_heatmaps_tupless, xs, ys = speaker.generate_all_heatmaps(scene, step=step)
    all_heatmaps_tuples = all_heatmaps_tupless[0]
    x = np.array( [list(xs-step*0.5)]*len(ys) )
    y = np.array( [list(ys-step*0.5)]*len(xs) ).T

    # all_heatmaps_tuples = []
    # for lmk, d in all_heatmaps_dict.items():
    #     for rel, heatmaps in d.items():
    #         all_heatmaps_tuples.append( (lmk,rel,heatmaps) )
    # all_heatmaps_tuples = all_heatmaps_tuples[:100]
    lmks, rels, heatmapss = zip(*all_heatmaps_tuples)
    graphmax1 = graphmax2 = 0
    meanings = zip(lmks,rels)
    landmarks = list(set(lmks))
    relations = list(set(rels))

    demo_sentences = ['near to the left edge of the table',
                      'somewhat near to the right edge of the table',
                      'on the table',
                      'on the middle of the table',
                      'at the lower left corner of the table',
                      'far from the purple prism']

    epsilon = 0.0001
    def heatmaps_for_sentence(sentence, iteration, good_meanings, good_heatmapss, graphmax1, graphmax2):

        posteriors = get_all_sentence_posteriors(sentence, good_meanings, printing=printing)
        # print sorted(zip(posteriors, meanings))
        # posteriors /= posteriors.sum()
        # for p,(l,r) in sorted(zip(posteriors, good_meanings)):
        #     print p, l, l.ori_relations, r, (r.distance, r.measurement.best_degree_class, r.measurement.best_distance_class ) if hasattr(r,'measurement') else 'No measurement'
        big_heatmap1 = None
        big_heatmap2 = None
        for m,(h1,h2) in zip(good_meanings, good_heatmapss):
            lmk,rel = m
            p = posteriors[rel]*posteriors[lmk]
            graphmax1 = max(graphmax1,h1.max())
            graphmax2 = max(graphmax2,h2.max())
            if big_heatmap1 is None:
                big_heatmap1 = p*h1
                big_heatmap2 = p*h2
            else:
                big_heatmap1 += p*h1
                big_heatmap2 += p*h2

        # good_meanings,good_heatmapss = zip(*[ (meaning,heatmaps) for posterior,meaning,heatmaps in zip(posteriors,good_meanings,good_heatmapss) if posterior > epsilon])

#        print big_heatmap1.shape
#        print xs.shape, ys.shape

        plt.figure(iteration)
        plt.suptitle(sentence)
        plt.subplot(121)

        probabilities1 = big_heatmap1.reshape( (len(xs),len(ys)) ).T
        plt.pcolor(x, y, probabilities1, cmap = 'jet', edgecolors='none', alpha=0.7)#, vmin=0, vmax=0.02)


        for lmk in scene.landmarks.values():
            if isinstance(lmk.representation, GroupLineRepresentation):
                xx = [lmk.representation.line.start.x, lmk.representation.line.end.x]
                yy = [lmk.representation.line.start.y, lmk.representation.line.end.y]
                plt.fill(xx,yy,facecolor='none',linewidth=2)
            elif isinstance(lmk.representation, RectangleRepresentation):
                rect = lmk.representation.rect
                xx = [rect.min_point.x,rect.min_point.x,rect.max_point.x,rect.max_point.x]
                yy = [rect.min_point.y,rect.max_point.y,rect.max_point.y,rect.min_point.y]
                plt.fill(xx,yy,facecolor='none',linewidth=2)
                plt.text(rect.min_point.x+0.01,rect.max_point.y+0.02,lmk.name)

        plt.plot(speaker.location.x,
                 speaker.location.y,
                 'bx',markeredgewidth=2)

        plt.axis('scaled')
        plt.axis([scene_bb.min_point.x, scene_bb.max_point.x, scene_bb.min_point.y, scene_bb.max_point.y])
        plt.colorbar()
        plt.title('Likelihood of sentence given location(s)')

        plt.subplot(122)

        probabilities2 = big_heatmap2.reshape( (len(xs),len(ys)) ).T
        plt.pcolor(x, y, probabilities2, cmap = 'jet', edgecolors='none', alpha=0.7)#, vmin=0, vmax=0.02)

        for lmk in scene.landmarks.values():
            if isinstance(lmk.representation, GroupLineRepresentation):
                xx = [lmk.representation.line.start.x, lmk.representation.line.end.x]
                yy = [lmk.representation.line.start.y, lmk.representation.line.end.y]
                plt.fill(xx,yy,facecolor='none',linewidth=2)
            elif isinstance(lmk.representation, RectangleRepresentation):
                rect = lmk.representation.rect
                xx = [rect.min_point.x,rect.min_point.x,rect.max_point.x,rect.max_point.x]
                yy = [rect.min_point.y,rect.max_point.y,rect.max_point.y,rect.min_point.y]
                plt.fill(xx,yy,facecolor='none',linewidth=2)
                plt.text(rect.min_point.x+0.01,rect.max_point.y+0.02,lmk.name)

        plt.plot(speaker.location.x,
                 speaker.location.y,
                 'bx',markeredgewidth=2)

        plt.axis('scaled')
        plt.axis([scene_bb.min_point.x, scene_bb.max_point.x, scene_bb.min_point.y, scene_bb.max_point.y])
        plt.colorbar()
        plt.title('Likelihood of location(s) given sentence')
        plt.draw()
        plt.show()
        return good_meanings, good_heatmapss, graphmax1, graphmax2


    def loop(num_iterations):
        min_dists = []
        # max_dists = []
        # avg_min = []
        # max_mins = []
        golden_log_probs = []
        # avg_golden_log_probs = []
        golden_entropies = []
        # avg_golden_entropies = []
        golden_ranks = []
        # avg_golden_ranks = []
        epsilon = 1e-15
        for iteration in range(num_iterations):
            logger(('Iteration %d' % iteration),'okblue')
            rand_p = Vec2(random()*table.width+table.min_point.x, random()*table.height+table.min_point.y)
            meaning, sentence = generate_sentence(rand_p, consistent, scene, speaker, usebest=True, printing=printing)
            logger( 'Generated sentence: %s' % sentence)
            trajector = Landmark( 'point', PointRepresentation(rand_p), None, Landmark.POINT )
            if cheating:
                landmark, relation = meaning.args[0],meaning.args[3]
            else:
                if explicit_pointing:
                    landmark = meaning.args[0]
                if ambiguous_pointing:
                    pointing_point = landmark.representation.middle + Vec2(random()*0.1-0.05,random()*0.1-0.05)
                #_, bestsentence = generate_sentence(rand_p, consistent, scene, speaker, usebest=True, printing=printing)

                try:
                    golden_posteriors = get_all_sentence_posteriors(sentence, meanings, golden=True, printing=printing)
                except ParseError as e:
                    logger( e )
                    continue
                epsilon = 1e-15
                ps = [[golden_posteriors[lmk]*golden_posteriors[rel],(lmk,rel)] for lmk, rel in meanings if ((not explicit_pointing) or lmk == landmark)]

                temp = None
                for i,(p,(lmk,rel)) in enumerate(ps):
                    # lmk,rel = meanings[i]
                    # logger( '%f, %s' % (p, m2s(lmk,rel)))
                    head_on = speaker.get_head_on_viewpoint(lmk)
                    if not explicit_pointing:
                        ps[i][0] *= speaker.get_landmark_probability(lmk, landmarks, PointRepresentation(rand_p))[0]
                    if ambiguous_pointing:
                        ps[i][0] *= speaker.get_landmark_probability(lmk, landmarks, PointRepresentation(pointing_point))[0]
                    ps[i][0] *= speaker.get_probabilities_points( np.array([rand_p]), rel, head_on, lmk)[0]
                    if lmk == meaning.args[0] and rel == meaning.args[3]:
                        temp = i

                ps,_meanings = zip(*ps)
                print ps
                ps = np.array(ps)
                ps += epsilon
                ps = ps/ps.sum()
                temp = ps[temp]

                ps = sorted(zip(ps,_meanings),reverse=True)

                logger( 'Attempted to say: %s' %  m2s(meaning.args[0],meaning.args[3]) )
                logger( 'Interpreted as: %s' % m2s(ps[0][1][0],ps[0][1][1]) )
                logger( 'Attempted: %f vs Interpreted: %f' % (temp, ps[0][0]))

                # logger( 'Golden entropy: %f, Max entropy %f' % (golden_entropy, max_entropy))

                landmark, relation = ps[0][1]
            head_on = speaker.get_head_on_viewpoint(landmark)
            all_descs = speaker.get_all_meaning_descriptions(trajector, scene, landmark, relation, head_on, 1)

            distances = []
            for desc in all_descs:
                distances.append([edit_distance( sentence, desc ), desc])

            distances.sort()
            print distances

            correction = distances[0][1]
            accept_correction( meaning, correction, update_scale=scale, eval_lmk=(not explicit_pointing), printing=printing )

            def probs_metric():
                bestmeaning, bestsentence = generate_sentence(rand_p, consistent, scene, speaker, usebest=True, printing=printing)
                sampled_landmark, sampled_relation = bestmeaning.args[0], bestmeaning.args[3]
                golden_posteriors = get_all_sentence_posteriors(bestsentence, meanings, golden=True, printing=printing)
                ps = np.array([golden_posteriors[lmk]*golden_posteriors[rel] for lmk, rel in meanings])
                rank = None
                for i,p in enumerate(ps):
                    lmk,rel = meanings[i]
                    # logger( '%f, %s' % (p, m2s(lmk,rel)))
                    head_on = speaker.get_head_on_viewpoint(lmk)
                    ps[i] *= speaker.get_landmark_probability(lmk, landmarks, PointRepresentation(rand_p))[0]
                    ps[i] *= speaker.get_probabilities_points( np.array([rand_p]), rel, head_on, lmk)
                    if lmk == sampled_landmark and rel == sampled_relation:
                        idx = i

                ps += epsilon
                ps = ps/ps.sum()
                prob = ps[idx]
                rank = sorted(ps, reverse=True).index(prob)
                entropy = entropy_of_probs(ps)


                head_on = speaker.get_head_on_viewpoint(sampled_landmark)
                all_descs = speaker.get_all_meaning_descriptions(trajector, scene, sampled_landmark, sampled_relation, head_on, 1)
                distances = []
                for desc in all_descs:
                    distances.append([edit_distance( sentence, desc ), desc])
                distances.sort()
                return prob,entropy,rank,distances[0][0]

            try:
                prob,entropy,rank,ed = probs_metric()
            except ParseError as e:
                logger( e )
                continue

            golden_log_probs.append( prob )
            # avg_golden_log_probs.append( np.mean(golden_log_probs[-window:]) )

            golden_entropies.append( entropy )
            # avg_golden_entropies.append( np.mean(golden_entropies[-window:]) )

            golden_ranks.append( rank )
            # avg_golden_ranks.append( np.mean(golden_ranks[-window:]) )

            min_dists.append( ed )
            # avg_min.append( np.mean(min_dists[-window:]) )
            # max_mins.append( max(min_dists[-window:]) )

            # if iteration % 200 == 0:
            #     f = open('data.txt','a')
            #     f.write(str(datetime.now())+'\n')
            #     f.write('max_mins\n')
            #     f.write(str(max_mins)+'\n')
            #     f.write('golden_probs\n')
            #     f.write(str(golden_log_probs)+'\n')
        return zip(golden_log_probs, golden_entropies, golden_ranks, min_dists)

    num_processors = 1
    num_each = int(num_iterations/num_processors)
    num_iterationss = [num_each]*num_processors
    # num_iterationss[-1] += num_iterations-num_each*num_processors
    print num_iterationss
    lists = parmap(loop,num_iterationss)
    print num_processors, num_each
    print len(lists), len(lists[0])
    result = []
    for i in range(num_each):
        print i
        for j in range(num_processors):
            print '  ',j,len(lists),len(lists[j])
            result.append( lists[j][i] )

    golden_log_probs,golden_entropies,golden_ranks,min_dists = zip(*result)
    def running_avg(arr):
        return [np.mean(arr[i:i+window]) for i in range(len(arr)-window)]
    avg_golden_log_probs = running_avg(golden_log_probs)
    avg_golden_entropies = running_avg(golden_entropies)
    avg_golden_ranks = running_avg(golden_ranks)
    avg_min = running_avg(min_dists)


    if cheating:
        title = 'Cheating (Telepathy)'
    if explicit_pointing:
        title = 'Explicit Pointing\n(Telepath Landmark only)'
    if ambiguous_pointing:
        title = 'Ambiguous Pointing'
    plt.plot(avg_min, 'o-', color='RoyalBlue')
    # plt.plot(max_mins, 'x-', color='Orange')
    plt.ylabel('Edit Distance')
    plt.title(title)
    plt.show()
    plt.draw()

    plt.figure()
    plt.suptitle(title)
    plt.subplot(211)
    plt.plot(golden_log_probs, 'o-', color='RoyalBlue')
    plt.plot(avg_golden_log_probs, 'x-', color='Orange')
    plt.ylabel('Golden Probability')

    plt.subplot(212)
    plt.plot(golden_ranks, 'o-', color='RoyalBlue')
    plt.plot(avg_golden_ranks, 'x-', color='Orange')
    plt.ylim([0,max(avg_golden_ranks)+10])
    plt.ylabel('Golden Rank')
    plt.ioff()
    plt.show()
    plt.draw()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_iterations', type=int, default=1)
    parser.add_argument('-l', '--location', type=Point)
    parser.add_argument('-c','--cheating', action='store_true')
    parser.add_argument('-e','--explicit', action='store_true')
    parser.add_argument('-a','--ambiguous', action='store_true')
    parser.add_argument('--consistent', action='store_true')
    args = parser.parse_args()

    scene, speaker = construct_training_scene()

    autocorrect(scene, speaker, args.num_iterations, window=20, scale=1000, consistent=args.consistent,
        cheating=args.cheating, explicit_pointing=args.explicit, ambiguous_pointing=args.ambiguous)


