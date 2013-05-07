#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import utils
from utils import (get_meaning,
                   rel_type,
                   m2s,
                   get_lmk_ori_rels_str,
                   get_landmark_parent_chain,
                   logger)

from models import CProduction

from semantics.run import construct_training_scene

from semantics.representation import PointRepresentation
from semantics.landmark import Landmark

from myrandom import random
random = random.random
from planar import Vec2


# this class is only used for the --location command line argument
class Point(object):
    def __init__(self, s):
        x, y = s.split(',')
        self.xy = (float(x), float(y))
        self.x, self.y = self.xy

    def __repr__(self):
        return 'Point(%s, %s)' % self.xy


def generate_sentence(loc, scene, speaker, usebest=False, golden=False, printing=True, visualize=False):
    utils.scene = utils.ModelScene(scene, speaker)

    (lmk, _, _), (rel, _, _) = get_meaning(loc=loc, usebest=usebest)

    if visualize:
        trajector = Landmark( 'point', PointRepresentation(Vec2(*loc)), None, Landmark.POINT )
        head_on = speaker.get_head_on_viewpoint(lmk)
        speaker.visualize(scene, trajector, head_on, lmk, rel, '<empty>', 0.04)

    meaning1 = m2s(lmk, rel)
    logger(get_landmark_parent_chain(lmk))
    logger(meaning1)

    lmk_class = lmk.object_class
    lmk_ori_rels = get_lmk_ori_rels_str(lmk)
    lmk_color = lmk.color
    rel_class = rel_type(rel)
    dist_class = (rel.measurement.best_distance_class if hasattr(rel, 'measurement') else None)
    deg_class = (rel.measurement.best_degree_class if hasattr(rel, 'measurement') else None)

    columns = [
        CProduction.landmark_class,
        CProduction.landmark_orientation_relations,
        CProduction.landmark_color,
        CProduction.relation,
        CProduction.relation_distance_class,
        CProduction.relation_degree_class
    ]

    prods = CProduction.get_productions_context(relation=rel_class,
                                                relation_distance_class=dist_class,
                                                relation_degree_class=deg_class,
                                                landmark_class=lmk_class,
                                                landmark_color=lmk_color,
                                                landmark_orientation_relations=lmk_ori_rels,
                                                golden=golden)
    if len(prods) < 1:
        print 'No production for context (%s, %s, %s, %s, %s, %s)' % (rel_class, dist_class, deg_class, lmk_class, lmk_color, lmk_ori_rels)

    for p in prods:
        print '[%d] %s -> %s p=[%s] lmk=[%s %s %s] rel=[%s %s %s]' % (p.count, p.lhs, p.rhs, p.parent, p.landmark_class, p.landmark_color, p.landmark_orientation_relations, p.relation, p.relation_degree_class, p.relation_distance_class)

    sorted_map = {}

    for col in columns:
        sorted_map[str(col)] = []

    lmk_prod = lmk_color_prod = lmk_ori_rels_prod = rel_prod = rel_deg_prod = rel_dist_prod = ''

    # find all equivalence clases

    for prod in prods:
        print '%s -> %s [%s]' % (prod.lhs, prod.rhs, prod.parent)
        ratios = []

        for col in columns:
            ratio = CProduction.get_entropy_ratio_sample_dependent(lhs=prod.lhs,
                                                                   rhs=prod.rhs,
                                                                   column=col,
                                                                   golden=golden,
                                                                   verbose=False)

            # ratio = CProduction.get_entropy_ratio_full_context(lhs=prod.lhs,
            #                                                    rhs=prod.rhs,
            #                                                    column=col,
            #                                                    parent=prod.parent,
            #                                                    lmk_class=prod.landmark_class,
            #                                                    lmk_ori_rels=prod.landmark_orientation_relations,
            #                                                    lmk_color=prod.landmark_color,
            #                                                    rel=prod.relation,
            #                                                    dist_class=prod.relation_distance_class,
            #                                                    deg_class=prod.relation_degree_class,
            #                                                    golden=golden,
            #                                                    verbose=False)

            ratios.append( (ratio, col) )
            sorted_map[str(col)].append( (ratio, prod.lhs, prod.rhs, prod.parent) )

        for ratio, col in sorted(ratios):
            print '\t%f --- %s' % (ratio, col)

        print '\n\n'

    for c in sorted_map:
        sorted_map[str(c)] = sorted(sorted_map[str(c)])

    for k in sorted_map:
        print k
        for ratio,lhs,rhs,parent in sorted_map[k]:
            print '\t[%f] %s -> %s [%s]' % (ratio,lhs,rhs,parent)

    expanded_prods = set()

    def expand_production(production_str, parent=None):
        result = []
        parts = production_str.split()
        for p in parts:
            # if p[0] in 'SEP':
            if p.upper() == p and (p+parent) not in expanded_prods:
                expanded_prods.add(p+parent)
                prods = CProduction.get_productions_context(lhs=p,
                                                            parent=parent,
                                                            relation=rel_class,
                                                            relation_distance_class=dist_class,
                                                            relation_degree_class=deg_class,
                                                            landmark_class=lmk_class,
                                                            landmark_color=lmk_color,
                                                            landmark_orientation_relations=lmk_ori_rels,
                                                            golden=golden)
                print '\t%s [%s]:' % (p, parent)
                for prod in prods:
                    print '\t\t[%d] %s -> %s' % (prod.count, prod.lhs, prod.rhs)
                    result.append( [prod.lhs, prod.rhs, prod.parent, prod.count] )
                    result.extend( expand_production(prod.rhs, prod.lhs) )


        return result

    def find_production(meaning_part):
        prefixes = ['E', 'P', 'S']
        ratio_thresh = 1.5
        prod_score = 0.0

        for ratio,lhs,rhs,parent in sorted_map[str(meaning_part)]:
            found = False
            for pfx in prefixes:
                # if lhs.startswith(pfx) and ratio <= ratio_thresh:
                if lhs.upper() == lhs and ratio <= ratio_thresh:
                    prod = lhs + '->' + rhs
                    prod_score = ratio
                    expand_production(rhs, lhs)
                    found = True
                    break
            if found: break
        else:
            prod = '<empty>'

        return prod, prod_score

    # find productions for all aprts of our meaning
    print '\n\n'
    print 'Production Expansions:'
    lmk_prod, s1 = find_production(CProduction.landmark_class)
    lmk_color_prod, s2 = find_production(CProduction.landmark_color)
    lmk_ori_rels_prod, s3 = find_production(CProduction.landmark_orientation_relations)
    rel_prod, s4 = find_production(CProduction.relation)
    rel_deg_prod, s5 = find_production(CProduction.relation_degree_class)
    rel_dist_prod, s6 = find_production(CProduction.relation_distance_class)
    print '\n\n'

    sentence =  '\nlmk = %s [%f]\nlmk_color = %s [%f]\nlmk_ori = %s [%f]\nrel = %s [%f]\nrel_deg = %s [%f]\nrel_dist = %s [%f]' % \
        (lmk_prod, s1, lmk_color_prod, s2, lmk_ori_rels_prod, s3, rel_prod, s4, rel_deg_prod, s5, rel_dist_prod, s6)

    return meaning1, sentence



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--location', type=Point)
    parser.add_argument('-b', '--best', action='store_true')
    parser.add_argument('-g', '--golden', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-n', '--num-sentences', type=int, default=1)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    if not args.location:
        args.location = Point(str(random()*0.8-0.4)+','+str(random()*0.6+0.4))

    print 'Location: %s' % args.location

    scene, speaker = construct_training_scene()

    for _ in range(args.num_sentences):
        _, sentence = generate_sentence(args.location.xy, scene, speaker, usebest=args.best, golden=args.golden, printing=args.verbose, visualize=args.visualize)
        logger('Generated sentence: %s' % sentence)
        print '\n\n'
