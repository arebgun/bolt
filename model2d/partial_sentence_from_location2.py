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
import shelve


# this class is only used for the --location command line argument
class Point(object):
    def __init__(self, s):
        x, y = s.split(',')
        self.xy = (float(x), float(y))
        self.x, self.y = self.xy

    def __repr__(self):
        return 'Point(%s, %s)' % self.xy


columns = [
    CProduction.landmark_class,
    CProduction.landmark_orientation_relations,
    CProduction.landmark_color,
    CProduction.relation,
    CProduction.relation_distance_class,
    CProduction.relation_degree_class
]


def generate_sentence(loc, scene, speaker, entropies, usebest=False, golden=False, printing=True, visualize=False):
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

    col2val = {
        CProduction.landmark_class: lmk_class,
        CProduction.landmark_orientation_relations: lmk_ori_rels,
        CProduction.landmark_color: lmk_color,
        CProduction.relation: rel_class,
        CProduction.relation_distance_class: dist_class,
        CProduction.relation_degree_class: deg_class,
    }

    prods = {}
    for col in columns:
        for score,rhs,lhs,parent in entropies[str(col)][:int(0.1*len(entropies[str(col)]))]:
            # p = '%s -> %s [%s]' % (rhs,lhs,parent)
            p = (rhs,lhs,parent)
            if p not in prods: prods[p] = []
            prods[p].append( (col, score) )

    sorted_map = {}
    for p in prods:
        print '%s -> %s [%s]' % p

        for col,score in prods[p]:
            print '\t%s [%f]' % (str(col), score)

        lhs,rhs,parent = p
        for column,entropy in prods[p]:
            prob = CProduction.get_probability(lhs, rhs, column, col2val[column], parent)
            if str(column) not in sorted_map: sorted_map[str(column)] = []
            sorted_map[str(column)].append( (prob, lhs, rhs, parent, entropy) )
            # print lhs,rhs,parent,str(column),col2val[column], prob

    for r in sorted_map:
        sorted_map[r].sort(reverse=True)

    for k in sorted_map:
        print k
        for prob,lhs,rhs,parent,entropy in sorted_map[k]:
            print '\t[%f] %s -> %s [%s] %f' % (prob,lhs,rhs,parent,entropy)


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
        ratio_thresh = 0
        prod_score = 0.0

        for ratio,lhs,rhs,parent,entropy in sorted_map[str(meaning_part)]:
            found = False
            for pfx in prefixes:
                # if lhs.startswith(pfx) and ratio <= ratio_thresh:
                if lhs.upper() == lhs and ratio > ratio_thresh:
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
    rel_deg_prod, s5 = find_production(CProduction.relation_degree_class) if deg_class is not None else ('<empty>', 0.0)
    rel_dist_prod, s6 = find_production(CProduction.relation_distance_class) if deg_class is not None else ('<empty>', 0.0)
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

    f = shelve.open('entropies_trained_grouped_sample_dep.shelf')
    ents = f['entropies']
    f.close()

    if not args.location:
        args.location = Point(str(random()*0.8-0.4)+','+str(random()*0.6+0.4))

    print 'Location: %s' % args.location

    scene, speaker = construct_training_scene()

    for _ in range(args.num_sentences):
        _, sentence = generate_sentence(args.location.xy, scene, speaker, ents, usebest=args.best, golden=args.golden, printing=args.verbose, visualize=args.visualize)
        logger('Generated sentence: %s' % sentence)
        print '\n\n'
