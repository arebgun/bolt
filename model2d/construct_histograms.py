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
from semantics.relation import Degree, Measurement

from myrandom import random
random = random.random
from planar import Vec2
import shelve

from matplotlib.pyplot import figure, show
import numpy as npy
from collections import Counter
from random import choice


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


def generate_sentence(trajector, scene, speaker, entropies, golden=False, printing=True, visualize=False, meaning=None):
    if meaning is not None:
        lmk, rel = meaning
        head_on = speaker.get_head_on_viewpoint(lmk)
    else:
        lmk, rel, head_on = speaker.sample_meaning(trajector, scene, 1)

    if visualize:
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
        for score,lhs,rhs,parent in entropies[str(col)][:int(0.25*len(entropies[str(col)]))]:
            # p = '%s -> %s [%s]' % (rhs,lhs,parent)
            p = (lhs,rhs,parent)

            if p not in prods: prods[p] = []
            prods[p].append( (col, score) )

    sorted_map = {}
    for p in prods:
        print '%s -> %s [%s]' % p

        for col,score in prods[p]:
            print '\t%s [%f]' % (str(col), score)

        lhs,rhs,parent = p
        for column,entropy in prods[p]:
            prob = CProduction.get_probability(lhs, rhs, column, col2val[column], parent, golden=golden)# * (1.0/entropy)
            if str(column) not in sorted_map: sorted_map[str(column)] = []
            sorted_map[str(column)].append( (prob, lhs, rhs, parent, entropy) )
            # print lhs,rhs,parent,str(column),col2val[column], prob

    for r in sorted_map:
        sorted_map[r].sort(reverse=True)

    for k in sorted_map:
        print k
        s = zip(*sorted_map[k])

        fig = figure()
        ax1 = fig.add_subplot(111)
        col = ax1.scatter(s[0], s[4])
        fig.savefig('%s_%s.png' % (k, 'golden' if golden else 'trained'))

        # show()

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

    # find productions for all parts of our meaning
    print '\n\n'
    print 'Production Expansions:'
    lmk_prod, s1 = find_production(CProduction.landmark_class)
    lmk_color_prod, s2 = find_production(CProduction.landmark_color) if lmk_color is not None else ('<not applicable>', 0.0)
    lmk_ori_rels_prod, s3 = find_production(CProduction.landmark_orientation_relations) if lmk_ori_rels is not None else ('<not applicable>', 0.0)
    rel_prod, s4 = find_production(CProduction.relation)
    rel_deg_prod, s5 = find_production(CProduction.relation_degree_class) if (deg_class is not None and deg_class != Degree.NONE) else ('<not applicable>', 0.0)
    rel_dist_prod, s6 = find_production(CProduction.relation_distance_class) if (dist_class is not None and dist_class != Measurement.NONE) else ('<not applicable>', 0.0)
    print '\n\n'

    sentence =  '\nlmk = %s [%f]\nlmk_color = %s [%f]\nlmk_ori = %s [%f]\nrel = %s [%f]\nrel_deg = %s [%f]\nrel_dist = %s [%f]' % \
        (lmk_prod, s1, lmk_color_prod, s2, lmk_ori_rels_prod, s3, rel_prod, s4, rel_deg_prod, s5, rel_dist_prod, s6)

    return (lmk,rel), sentence, sorted_map

def calculate_entropies():
    group_by_parent = True
    sorted_map = Counter()

    # get all unique productions of the form lhs -> rhs
    unique_prods = CProduction.get_unique_productions(group_by_rhs=True, group_by_parent=group_by_parent, golden=False)

    for prod in unique_prods:
        ratios = []

        for col in columns:
            ratio = CProduction.get_entropy_ratio_sample_dependent(lhs=prod.lhs, rhs=prod.rhs, column=col, golden=False, verbose=False)
            # ratio = CProduction.get_mutual_information(lhs=prod.lhs, rhs=prod.rhs, column=col, golden=golden, verbose=verbose)
            ratios.append( (ratio, col) )
            if str(col) not in sorted_map: sorted_map[str(col)] = []
            sorted_map[str(col)].append( (ratio, prod.lhs, prod.rhs, prod.parent) )

    for c in sorted_map:
        sorted_map[str(c)] = sorted(sorted_map[str(c)])

    return sorted_map

def calculate_score(ents):
    scores = Counter()

    for col,l in ents.items():
        print col, len(l)
        for freq,lhs,rhs,parent,ent in l:
            if freq > 30:
                print '\t', lhs,rhs,parent,ent,freq, ent/freq

        col_score_total = sum([ent/freq for freq,_,_,_,ent in l if freq > 30])
        scores[col] = col_score_total / len(l)

    return scores


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--best', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-c', '--calculate-entropy', action='store_true')
    parser.add_argument('-f', '--entropy-file')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    # for each scenario
    #    for each semantic feature
    #       there will ditributuion over frequency and distribution over mutual information scores,
    #       we want a histrogram of both

    fn_golden = 'entropies_golden_grouped_ratio_sample_dep_200k.shelf'

    if args.entropy_file:
        fn_trained = args.entropy_file
    else:
        fn_trained = 'entropies_trained_grouped_ratio_sample_dep_1k.shelf'

    f = shelve.open(fn_golden)
    ents_golden = f['entropies']
    f.close()

    if args.calculate_entropy:
        ents_trained = calculate_entropies()
        f = shelve.open('tmp_entropies.shelf')
        f['entropies'] = ents_trained
        f.close()
    else:
        f = shelve.open(fn_trained)
        ents_trained = f['entropies']
        f.close()

    min_objects = 1
    max_objects = 7
    random_scene = True

    num_objects = int(random() * (max_objects - min_objects) + min_objects)
    scene, speaker = construct_training_scene(random=random_scene, num_objects=num_objects)
    utils.scene = utils.ModelScene(scene, speaker)

    table = scene.landmarks['table'].representation.rect
    t_min = table.min_point
    t_max = table.max_point
    t_w = table.width
    t_h = table.height

    all_objects = [lmk for lmk in scene.landmarks.values() if lmk.name != 'table']
    trajector = choice(all_objects)

    logger('Teacher chooses [%s]' % trajector)

    (lmk,rel), sentence_golden, tmap = generate_sentence(
        trajector,
        scene,
        speaker,
        ents_golden,
        golden=True,
        printing=args.verbose,
        visualize=args.visualize
    )

    print '\n\n', '*'*80, '\n\n\n'

    _, sentence_trained, smap = generate_sentence(
        trajector,
        scene,
        speaker,
        ents_trained,
        golden=False,
        printing=args.verbose,
        visualize=args.visualize,
        meaning=(lmk,rel)
    )

    print '\n\n', '*'*80, '\n\n\n'

    teach_score = calculate_score(tmap)
    print '\n\n', '*'*80, '\n\n\n'
    stud_score = calculate_score(smap)
    print '\n\n', '*'*80, '\n\n\n'

    logger('Generated Teacher sentence: %s' % sentence_golden)
    logger('Current teacher score: %s' % str(teach_score))
    print '\n\n', '*'*80, '\n\n\n'
    logger('Generated Student sentence: %s' % sentence_trained)
    logger('Current Student score: %s' % str(stud_score))
