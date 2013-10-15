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
from random import choice
from collections import Counter


columns = [
    CProduction.landmark_class,
    CProduction.landmark_orientation_relations,
    CProduction.landmark_color,
    CProduction.relation,
    CProduction.relation_distance_class,
    CProduction.relation_degree_class
]


def calculate_score(ents):
    scores = Counter()

    for col,l in ents.items():
        # for freq,lhs,rhs,parent,ent in l:
        #     if freq > 0:
        #         col_score_total += (ent/freq)

        col_score_total = sum([ent/freq for freq,_,_,_,ent in l if freq > 30])
        scores[col] = col_score_total / len(l)

    return scores


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


def generate_sentence(trajector, scene, speaker, entropies, usebest=False, golden=False, printing=True, visualize=False, meaning=None):
    #(lmk, _, _), (rel, _, _) = get_meaning(loc=loc, usebest=usebest)
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

            try:
                prods[p].append( (col, score) )
            except:
                prods[p] = [ (col, score) ]

    sorted_map = Counter()
    for p in prods:
        if printing:
            print '%s -> %s [%s]' % p

            for col,score in prods[p]:
                print '\t%s [%f]' % (str(col), score)

        lhs,rhs,parent = p
        for column,entropy in prods[p]:
            prob = CProduction.get_probability(lhs, rhs, column, col2val[column], parent, golden=golden)# * (1.0/entropy)

            try:
                sorted_map[str(column)].append( (prob, lhs, rhs, parent, entropy) )
            except:
                sorted_map[str(column)] = [ (prob, lhs, rhs, parent, entropy) ]

            # if printing: print lhs,rhs,parent,str(column),col2val[column], prob

    for r in sorted_map:
        sorted_map[r].sort(reverse=True)

    if printing:
        for k in sorted_map:
            print k
            # s = zip(*sorted_map[k])

            # fig = figure()
            # ax1 = fig.add_subplot(111)
            # col = ax1.scatter(s[0], s[4])
            # fig.savefig(k + '.png')
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
                # print '\t%s [%s]:' % (p, parent)
                for prod in prods:
                    # print '\t\t[%d] %s -> %s' % (prod.count, prod.lhs, prod.rhs)
                    result.append( [prod.lhs, prod.rhs, prod.parent, prod.count] )
                    result.extend( expand_production(prod.rhs, prod.lhs) )


        return result

    def find_production(meaning_part):
        prefixes = ['E', 'P', 'S']
        prod = '<empty'
        ratio_thresh = 0
        prod_score = 0.0
        expanded_prods = []

        if str(meaning_part) in sorted_map:
            for ratio,lhs,rhs,parent,entropy in sorted_map[str(meaning_part)]:
                found = False
                for pfx in prefixes:
                    # if lhs.startswith(pfx) and ratio <= ratio_thresh:
                    if lhs.upper() == lhs and ratio > ratio_thresh:
                        expanded_prods.append( [lhs, rhs, parent, ratio] )
                        prod = lhs + '->' + rhs + '[' + str(parent) + ']'
                        prod_score = ratio
                        expanded_prods.extend( expand_production(rhs, lhs) )
                        found = True
                        break
                if found: break

        return prod, prod_score, expanded_prods

    # find productions for all parts of our meaning
    # print '\n\n'
    # print 'Production Expansions:'
    lmk_prod, s1, ep1 = find_production(CProduction.landmark_class)
    lmk_color_prod, s2, ep2 = find_production(CProduction.landmark_color) if lmk_color is not None else ('<not applicable>', 0.0, [])
    lmk_ori_rels_prod, s3, ep3 = find_production(CProduction.landmark_orientation_relations) if lmk_ori_rels is not None else ('<not applicable>', 0.0, [])
    rel_prod, s4, ep4 = find_production(CProduction.relation)
    rel_deg_prod, s5, ep5 = find_production(CProduction.relation_degree_class) if (deg_class is not None and deg_class != Degree.NONE) else ('<not applicable>', 0.0, [])
    rel_dist_prod, s6, ep6 = find_production(CProduction.relation_distance_class) if (dist_class is not None and dist_class != Measurement.NONE) else ('<not applicable>', 0.0, [])
    # print '\n\n'

    sentence =  '\nlmk = %s [%f]\nlmk_color = %s [%f]\nlmk_ori = %s [%f]\nrel = %s [%f]\nrel_deg = %s [%f]\nrel_dist = %s [%f]' % \
        (lmk_prod, s1, lmk_color_prod, s2, lmk_ori_rels_prod, s3, rel_prod, s4, rel_deg_prod, s5, rel_dist_prod, s6)

    bow = Counter(' '.join([rel_deg_prod, rel_dist_prod, rel_prod, lmk_ori_rels_prod, lmk_color_prod, lmk_prod]).split())
    #expanded_prods = ep1 + ep2 + ep3 + ep4 + ep5 + ep6
    expanded_prods = {}
    expanded_prods[CProduction.landmark_class] = ep1
    expanded_prods[CProduction.landmark_color] = ep2
    expanded_prods[CProduction.landmark_orientation_relations] = ep3
    expanded_prods[CProduction.relation] = ep4
    expanded_prods[CProduction.relation_degree_class] = ep5
    expanded_prods[CProduction.relation_distance_class] = ep6

    return sentence, lmk, rel, bow, expanded_prods, sorted_map


# Generate a scene with n objects
# Teacher picks an object to be described
# Student generates a sentence, or, rather, chunks of sentence corresponding to the 6 semantic features
# Teacher generates its own set of chunks
# Compare chunks to figure out which chunks the student got right (need to figure out some kind of distance measure)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--best', action='store_true')
    parser.add_argument('-g', '--golden', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    fn_golden = 'entropies_golden_grouped_ratio_sample_dep_200k.shelf'
    fn_trained = 'entropies_trained_grouped_ratio_sample_dep_1k.shelf'

    logger('Reading teacher database entropy values from %s.' % str(fn_golden).upper(), 'okgreen')
    f = shelve.open(fn_golden)
    teach_ents = f['entropies']
    f.close()

    # print 'Student entropy values:', str(fn_trained).upper()
    # f = shelve.open(fn_trained)
    # stud_ents = f['entropies']
    # f.close()

    min_objects = 1
    max_objects = 7
    random_scene = True

    # all scores added, divided by number of iterations to get the average
    scores = Counter()

    # want to store by meaning to compare performance for a single meaning
    raw_scores = []
    teach_raw_scores = []

    # want to store all scores to compare avergae performance across all meanings
    raw_scores_by_meaning = {}

    iterations = 500
    batch = 10
    num_per_scene = 20

    # logger('Calculating student database entropies...', 'okgreen')
    # stud_ents = calculate_entropies()
    # logger('Done.')

    for i in range(iterations):
        # if i > 0 and (i % batch == 0 or i == (iterations-1)):
        logger('Calculating student database entropies', 'okgreen')
        stud_ents = calculate_entropies()

        f = shelve.open('learning_data.shelf')
        f['scores'] = scores
        f['raw_scores'] = raw_scores
        f['raw_scores_by_meaning'] = raw_scores_by_meaning
        f['teach_raw_scores'] = teach_raw_scores
        f.close()

        logger('Scene %d' % i, 'okblue')

        # Generate a scene with n objects
        num_objects = int(random() * (max_objects - min_objects) + min_objects)
        scene, speaker = construct_training_scene(random=random_scene, num_objects=num_objects)
        utils.scene = utils.ModelScene(scene, speaker)

        table = scene.landmarks['table'].representation.rect
        t_min = table.min_point
        t_max = table.max_point
        t_w = table.width
        t_h = table.height

        for j in range(num_per_scene):
            # Teacher picks an object to be described
            all_objects = [lmk for lmk in scene.landmarks.values() if lmk.name != 'table']
            trajector = choice(all_objects)

            logger('Object %d' % j, 'okblue')
            logger('Teacher chooses [%s]' % trajector)

            # Teacher generates its own set of chunks
            teach_sent, tlmk, trel, tbow, expanded_prods, tmap = generate_sentence(
                trajector,
                scene,
                speaker,
                teach_ents,
                usebest=False,
                golden=True,
                printing=False,
                visualize=False)

            logger( 'Teacher generated meaning: %s' % m2s(tlmk, trel) )
            # logger( 'Teacher bag of words: %s' % str(tbow) )
            # logger( 'Teacher expanded productions set: %s' % str(expanded_prods) )
            # logger('Teacher generated sentence: %s' % teach_sent)
            # print '\n\n', '*'*80, '\n\n\n'

            logger('Calculating teacher score', 'okgreen')
            teach_score = calculate_score(tmap)
            logger('Current teacher score: %s' % str(teach_score))

            if len(stud_ents) > 0:
                # Student generates a sentence, or, rather, chunks of sentence corresponding to the 6 semantic features
                stud_sent, _, _, sbow, _, smap = generate_sentence(
                    trajector,
                    scene,
                    speaker,
                    stud_ents,
                    usebest=False,
                    golden=False,
                    printing=False,
                    visualize=False,
                    meaning=(tlmk, trel))

                # logger( 'Student generated meaning: %s' % m2s(slmk, srel) )
                # logger( 'Student bag of words: %s' % str(sbow) )
                # logger('Student generated sentence: %s' % stud_sent)
                # print '\n\n', '*'*80, '\n\n\n'

                logger('Calculating student score', 'okgreen')
                stud_score = calculate_score(smap)
            else:
                stud_score = Counter()
                for col in columns:
                    stud_score[str(col)] = 1

            logger('Current student score: %s' % str(stud_score))
            scores += stud_score
            raw_scores.append(stud_score)
            teach_raw_scores.append(teach_score)

            try:
                raw_scores_by_meaning[m2s(tlmk,trel)].append(stud_score)
            except:
                raw_scores_by_meaning[m2s(tlmk,trel)] = [stud_score]

            # update
            lmk_class = tlmk.object_class
            lmk_ori_rels = get_lmk_ori_rels_str(tlmk)
            lmk_color = tlmk.color
            rel_class = rel_type(trel)
            dist_class = (trel.measurement.best_distance_class if hasattr(trel, 'measurement') else None)
            deg_class = (trel.measurement.best_degree_class if hasattr(trel, 'measurement') else None)

            col2val = {
                CProduction.landmark_class: {'lmk_class': lmk_class},
                CProduction.landmark_orientation_relations: {'lmk_ori_rels': lmk_ori_rels},
                CProduction.landmark_color: {'lmk_color': lmk_color},
                CProduction.relation: {'rel': rel_class},
                CProduction.relation_distance_class: {'dist_class': dist_class},
                CProduction.relation_degree_class: {'deg_class': deg_class},
            }

            update = 5.1

            for col,prods in expanded_prods.items():
                for p in prods:
                    lhs, rhs, parent, _ = p
                    # print 'updating', lhs, '->', rhs, '[', parent, ']', 'for', str(col)

                    CProduction.update_production_counts(
                        update=update,
                        lhs=lhs,
                        rhs=rhs,
                        parent=parent,
                        lmk_class=lmk_class,
                        lmk_ori_rels=lmk_ori_rels,
                        lmk_color=lmk_color,
                        rel=rel_class,
                        dist_class=dist_class,
                        deg_class=deg_class,
                        multiply=False
                    )

                    CProduction.update_production_counts(
                        update=update,
                        lhs=lhs,
                        rhs=rhs,
                        parent=parent,
                        **col2val[col]
                    )

                    # remove double update
                    CProduction.update_production_counts(
                        update=-update,
                        lhs=lhs,
                        rhs=rhs,
                        parent=parent,
                        lmk_class=lmk_class,
                        lmk_ori_rels=lmk_ori_rels,
                        lmk_color=lmk_color,
                        rel=rel_class,
                        dist_class=dist_class,
                        deg_class=deg_class,
                        multiply=False
                    )


        if len(stud_ents) == 0:
            logger('Calculating student database entropies...', 'okgreen')
            calculate_entropies()

    print scores