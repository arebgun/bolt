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
from parse import get_modparse
from nltk.tree import ParentedTree

from semantics.run import construct_training_scene
from semantics.landmark import Landmark
from semantics.representation import PointRepresentation, LineRepresentation, RectangleRepresentation, GroupLineRepresentation
from nltk.metrics.distance import edit_distance
from planar import Vec2
from utils import logger, m2s
import numpy as np
from matplotlib import pyplot as plt

from location_from_sentence import get_all_sentence_posteriors

def autocorrect(scene, speaker, num_iterations=1, window=10, scale=1000, consistent=False):
    plt.ion()

    printing=False

    scene_bb = scene.get_bounding_box()
    scene_bb = scene_bb.inflate( Vec2(scene_bb.width*0.5,scene_bb.height*0.5) )
    table = scene.landmarks['table'].representation.get_geometry()

    min_dists = []
    max_dists = []
    avg_min = []
    max_mins = []
    golden_log_probs = []
    avg_golden_log_probs = []
    golden_entropies = []
    avg_golden_entropies = []

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

    for iteration in range(num_iterations):

        if iteration % 10 == 0:
            # for sentence in demo_sentences[:1]:
            # print 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX len(meanings)', len(meanings)
            #meanings, heatmapss, graphmax1, graphmax2 = heatmaps_for_sentence(demo_sentences[0], iteration, meanings, heatmapss, graphmax1, graphmax2)
            # print 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX len(meanings)', len(meanings)
            pass

        # for p,h in zip(posteriors, heatmaps):
        #     probabilities = h.reshape( (len(xs),len(ys)) ).T
        #     plt.pcolor(x, y, probabilities, cmap = 'jet', edgecolors='none', alpha=0.7)
        #     plt.colorbar()
        #     plt.show()


        logger('Iteration %d' % iteration)
        rand_p = Vec2(random()*table.width+table.min_point.x, random()*table.height+table.min_point.y)
        meaning, sentence = generate_sentence(rand_p, consistent, scene, speaker, printing=printing)
        bestmeaning, bestsentence = generate_sentence(rand_p, consistent, scene, speaker, usebest=True, printing=printing)

        logger( 'Generated sentence: %s' % sentence)
        logger( 'Best sentence: %s' % bestsentence)

        golden_posteriors = get_all_sentence_posteriors(sentence, meanings, golden=True, printing=printing)
        # epsilon = 1e-15
        # ps = np.array([golden_posteriors[lmk]*golden_posteriors[rel] for lmk, rel in meanings])

        def probs_metric():
            ps = np.array([golden_posteriors[lmk]*golden_posteriors[rel] for lmk, rel in meanings])
            temp = None
            for i,p in enumerate(ps):
                lmk,rel = meanings[i]
                # logger( '%f, %s' % (p, m2s(lmk,rel)))
                head_on = speaker.get_head_on_viewpoint(lmk)
                ps[i] *= speaker.get_landmark_probability(lmk, landmarks, PointRepresentation(rand_p))[0]
                ps[i] *= speaker.get_probabilities_points( np.array([rand_p]), rel, head_on, lmk)
                if lmk == meaning.args[0] and rel == meaning.args[3]:
                    temp = i

            ps += epsilon
            ps = ps/ps.sum()
            prob = ps[temp]
            entropy = entropy_of_probs(ps)
            return prob,entropy

        prob,entropy = probs_metric()
        golden_log_probs.append( prob )
        golden_entropies.append( entropy )
        avg_golden_log_probs.append( np.mean(golden_log_probs[-window:]) )
        avg_golden_entropies.append( np.mean(golden_entropies[-window:]) )

        # golden_log_probs.append( probs_metric(ps) )
        # avg_golden_log_probs.append( np.mean(golden_log_probs[-window:]) )

        trajector = Landmark( 'point', PointRepresentation(rand_p), None, Landmark.POINT )
        landmark, relation = meaning.args[0], meaning.args[3]
        head_on = speaker.get_head_on_viewpoint(landmark)
        all_descs = speaker.get_all_meaning_descriptions(trajector, scene, landmark, relation, head_on, 1)

        distances = []
        for desc in all_descs:
            distances.append([edit_distance( sentence, desc ), desc])

        distances.sort()
        print distances

        min_dists.append(distances[0][0])
        avg_min.append( np.mean(min_dists[-window:]) )
        max_mins.append( max(min_dists[-window:]) )

        correction = distances[0][1]
        accept_correction( meaning, correction, update_scale=scale, printing=printing )

        print np.mean(min_dists), avg_min, max_mins, golden_log_probs
        print;print

    plt.plot(avg_min, 'o-', color='RoyalBlue')
    plt.plot(max_mins, 'x-', color='Orange')
    plt.ylabel('Edit Distance')
    plt.title('Explicit Pointing\n (Telepathing Landmark only)')
    plt.show()
    plt.draw()

    plt.figure()
    plt.suptitle('Explicit Pointing\n (Telepathing Landmark only)')
    plt.subplot(211)
    plt.plot(golden_log_probs, 'o-', color='RoyalBlue')
    plt.plot(avg_golden_log_probs, 'x-', color='Orange')
    plt.ylabel('Golden Probability')

    plt.subplot(212)
    plt.plot(golden_entropies, 'o-', color='RoyalBlue')
    plt.plot(avg_golden_entropies, 'x-', color='Orange')
    plt.ylabel('Golden Entropy')
    plt.ioff()
    plt.show()
    plt.draw()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_iterations', type=int, default=1)
    parser.add_argument('-l', '--location', type=Point)
    parser.add_argument('--consistent', action='store_true')
    args = parser.parse_args()

    scene, speaker = construct_training_scene()

    autocorrect(scene, speaker, args.num_iterations, window=20, consistent=args.consistent)


