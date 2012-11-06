#!/usr/bin/env python
# coding: utf-8

from __future__ import division

from operator import itemgetter

import numpy as np
from nltk.tree import ParentedTree
from parse import get_modparse, ParseError

from utils import (
    parent_landmark,
    get_meaning,
    rel_type,
    m2s,
    count_lmk_phrases,
    logger,
    get_lmk_ori_rels_str,
    NONTERMINALS
)

from models import CProduction, CWord
from semantics.run import construct_training_scene

from semantics.representation import (
    GroupLineRepresentation,
    RectangleRepresentation
)

import matplotlib.pyplot as plt
from planar import Vec2
from itertools import product
import sys


def get_tree_probs(tree, lmk=None, rel=None, default_prob=0.001, default_ent=1000, golden=False, printing=True):

    lhs_rhs_parent_chain = []
    prob_chain = []
    entropy_chain = []
    term_prods = []

    lhs = tree.node

    if isinstance(tree[0], ParentedTree): rhs = ' '.join(n.node for n in tree)
    else: rhs = ' '.join(n for n in tree)

    if hasattr( tree.parent, 'node' ):
        parent = tree.parent.node if tree.parent else None
    else:
        parent = tree.parent().node if tree.parent() else None

    if lhs == 'RELATION':
        # everything under a RELATION node should ignore the landmark
        lmk = None

    if lhs == 'LANDMARK-PHRASE':
        # everything under a LANDMARK-PHRASE node should ignore the relation
        rel = None

    if lhs == parent == 'LANDMARK-PHRASE':
        # we need to move to the parent landmark
        lmk = parent_landmark(lmk)

    lmk_class = (lmk.object_class if lmk and lhs != 'LOCATION-PHRASE' else None)
    lmk_ori_rels = get_lmk_ori_rels_str(lmk) if lhs != 'LOCATION-PHRASE' else None
    lmk_color = (lmk.color if lmk and lhs != 'LOCATION-PHRASE' else None)
    rel_class = rel_type(rel) if lhs != 'LOCATION-PHRASE' else None
    dist_class = (rel.measurement.best_distance_class if hasattr(rel, 'measurement') and lhs != 'LOCATION-PHRASE' else None)
    deg_class = (rel.measurement.best_degree_class if hasattr(rel, 'measurement') and lhs != 'LOCATION-PHRASE' else None)

    if lhs in NONTERMINALS:
        cp_db = CProduction.get_production_counts(lhs=lhs,
                                                  parent=parent,
                                                  lmk_class=lmk_class,
                                                  lmk_ori_rels=lmk_ori_rels,
                                                  lmk_color=lmk_color,
                                                  rel=rel_class,
                                                  dist_class=dist_class,
                                                  deg_class=deg_class,
                                                  golden=golden)

        if cp_db.count() <= 0:
            if printing: logger('Could not expand %s (parent: %s, lmk_class: %s, lmk_ori_rels: %s, lmk_color: %s, rel: %s, dist_class: %s, deg_class: %s)' % (lhs, parent, lmk_class, lmk_ori_rels, lmk_color, rel_class, dist_class, deg_class))
            prob_chain.append( default_prob )
            entropy_chain.append( default_ent )
        else:
            ckeys, ccounts = zip(*[(cprod.rhs,cprod.count) for cprod in cp_db.all()])
            if printing: logger('Expanded %s (parent: %s, lmk_class: %s, lmk_ori_rels: %s, lmk_color: %s, rel: %s, dist_class: %s, deg_class: %s)' % (lhs, parent, lmk_class, lmk_ori_rels, lmk_color, rel_class, dist_class, deg_class))

            ccounter = {}
            for cprod in cp_db.all():
                if cprod.rhs in ccounter: ccounter[cprod.rhs] += cprod.count
                else: ccounter[cprod.rhs] = cprod.count + 1

            # we have never seen this RHS in this context before
            if rhs not in ccounter: ccounter[rhs] = 1

            ckeys, ccounts = zip(*ccounter.items())

            # add 1 smoothing
            ccounts = np.array(ccounts, dtype=float)
            ccount_probs = ccounts / ccounts.sum()
            cprod_entropy = -np.sum( (ccount_probs * np.log(ccount_probs)) )
            cprod_prob = ccounter[rhs]/ccounts.sum()

            # logger('ckeys: %s' % str(ckeys))
            # logger('ccounts: %s' % str(ccounts))
            # logger('rhs: %s, cprod_prob: %s, cprod_entropy: %s' % (rhs, cprod_prob, cprod_entropy))

            prob_chain.append( cprod_prob )
            entropy_chain.append( cprod_entropy )

        lhs_rhs_parent_chain.append( ( lhs, rhs, parent, lmk, rel ) )

        for subtree in tree:
            pc, ec, lrpc, tps = get_tree_probs(subtree, lmk, rel, default_prob, default_ent, golden=golden, printing=printing)
            prob_chain.extend( pc )
            entropy_chain.extend( ec )
            lhs_rhs_parent_chain.extend( lrpc )
            term_prods.extend( tps )

    else:
        cw_db = CWord.get_word_counts(pos=lhs,
                                      lmk_class=lmk_class,
                                      lmk_ori_rels=lmk_ori_rels,
                                      lmk_color=lmk_color,
                                      rel=rel_class,
                                      rel_dist_class=dist_class,
                                      rel_deg_class=deg_class,
                                      golden=golden)

        if cw_db.count() <= 0:
            # we don't know the probability or entropy values for the context we have never seen before
            # we just update the term_prods list
            if printing: logger('Could not expand %s (lmk_class: %s, lmk_ori_rels: %s, lmk_color: %s, rel: %s, dist_class: %s, deg_class: %s)' % (lhs, lmk_class, lmk_ori_rels, lmk_color, rel_class, dist_class, deg_class))
            prob_chain.append( default_prob )
            entropy_chain.append( default_ent )
        else:
            if printing: logger('Expanded %s (lmk_class: %s, lmk_ori_rels: %s, lmk_color: %s, rel: %s, dist_class: %s, deg_class: %s)' % (lhs, lmk_class, lmk_ori_rels, lmk_color, rel_class, dist_class, deg_class))

            ckeys, ccounts = zip(*[(cword.word,cword.count) for cword in cw_db.all()])

            ccounter = {}
            for cword in cw_db.all():
                if cword.word in ccounter: ccounter[cword.word] += cword.count
                else: ccounter[cword.word] = cword.count + 1

            # we have never seen this RHS in this context before
            if rhs not in ccounter: ccounter[rhs] = 1

            ckeys, ccounts = zip(*ccounter.items())

            # logger('ckeys: %s' % str(ckeys))
            # logger('ccounts: %s' % str(ccounts))

            # add 1 smoothing
            ccounts = np.array(ccounts, dtype=float)
            ccount_probs = ccounts/ccounts.sum()

            w_prob = ccounter[rhs]/ccounts.sum()
            w_entropy = -np.sum( (ccount_probs * np.log(ccount_probs)) )

            prob_chain.append(w_prob)
            entropy_chain.append(w_entropy)

        term_prods.append( (lhs, rhs, lmk, rel) )

    return prob_chain, entropy_chain, lhs_rhs_parent_chain, term_prods

def get_sentence_posteriors(sentence, iterations=1, extra_meaning=None, golden=False, printing=True):
    meaning_probs = {}
    # parse sentence with charniak and apply surgeries
    print 'parsing ...'
    _, modparse = get_modparse(sentence)
    t = ParentedTree.parse(modparse)
    print '\n%s\n' % t.pprint()
    num_ancestors = count_lmk_phrases(t) - 1

    for _ in xrange(iterations):
        (lmk, _, _), (rel, _, _) = get_meaning(num_ancestors=num_ancestors)
        meaning = m2s(lmk,rel)
        if meaning not in meaning_probs:
            ps = get_tree_probs(t, lmk, rel, golden=golden, printing=printing)[0]
            # print "Tree probs: ", zip(ps,rls)
            meaning_probs[meaning] = np.prod(ps)
        print '.'

    if extra_meaning:
        meaning = m2s(*extra_meaning)
        if meaning not in meaning_probs:
            ps = get_tree_probs(t, lmk, rel, golden=golden, printing=printing)[0]
            # print "Tree prob: ", zip(ps,rls)
            meaning_probs[meaning] = np.prod(ps)
        print '.'

    summ = sum(meaning_probs.values())
    for key in meaning_probs:
        meaning_probs[key] /= summ
    return meaning_probs.items()

def get_all_sentence_posteriors(sentence, meanings, golden=False, printing=True):

    print 'parsing ...'
    _, modparse = get_modparse(sentence)
    t = ParentedTree.parse(modparse)
    print '\n%s\n' % t.pprint()
    num_ancestors = count_lmk_phrases(t) - 1

    # logger(len(meanings))
    lmks, rels = zip(*meanings)
    lmks = set(lmks)
    rels = set(rels)
    # logger('num things ' + str(len(lmks))+' '+str(len(rels)))

    syms = ['\\', '|', '/', '-']
    sys.stdout.write('processing...\\')
    sys.stdout.flush()
    posteriors = {}
    for i,lmk in enumerate(lmks):
        if lmk.get_ancestor_count() != num_ancestors:
            p = 0
        else:
            ps = get_tree_probs(t[1], lmk, golden=golden, printing=printing)[0]
            p = np.prod(ps)
        posteriors[lmk] = p
        sys.stdout.write("\b%s" % syms[i % len(syms)])
        sys.stdout.flush()

    for i,rel in enumerate(rels):
        ps = get_tree_probs(t[0], rel=rel, golden=golden, printing=printing)[0]
        posteriors[rel] = np.prod(ps)
        sys.stdout.write("\b%s" % syms[i % len(syms)])
        sys.stdout.flush()

    for j in range(50):
        sys.stdout.write("\b.%s" % syms[(i+j) % len(syms)])
        sys.stdout.flush()
    print


    # for meaning in meanings:
    #     lmk,rel = meaning
    #     if lmk.get_ancestor_count() != num_ancestors:
    #         p = 0
    #     else:
    #         ps = get_tree_probs(t, lmk, rel, printing=False)[0]
    #         p = np.prod(ps)
    #     posteriors.append(p)
        # print p, lmk, lmk.ori_relations, rel, (rel.distance, rel.measurement.best_degree_class, rel.measurement.best_distance_class ) if hasattr(rel,'measurement') else 'No measurement'
    return posteriors


def get_sentence_meaning_likelihood(sentence, lmk, rel, printing=True):
    _, modparse = get_modparse(sentence)
    t = ParentedTree.parse(modparse)
    if printing: print '\n%s\n' % t.pprint()

    probs, entropies, lrpc, tps = get_tree_probs(t, lmk, rel, printing=printing)
    if np.prod(probs) == 0.0:
        logger('ERROR: Probability product is 0 for sentence: %s, lmk: %s, rel: %s, probs: %s' % (sentence, lmk, rel, str(probs)))
    return np.prod(probs), sum(entropies), lrpc, tps


def heatmaps_for_sentence(sentence, all_meanings, loi_infos, xs, ys, scene, speaker, step=0.02):
    printing=False
    scene_bb = scene.get_bounding_box()
    scene_bb = scene_bb.inflate( Vec2(scene_bb.width*0.5,scene_bb.height*0.5) )
    x = np.array( [list(xs-step*0.5)]*len(ys) )
    y = np.array( [list(ys-step*0.5)]*len(xs) ).T

    posteriors = get_all_sentence_posteriors(sentence, all_meanings, printing=printing)
    # posteriors_arr = np.array([posteriors[rel]*posteriors[lmk] for lmk,rel in all_meanings])
    # # print sorted(zip(posteriors, meanings))
    # posteriors_arr /= posteriors_arr.sum()
    # for p,(l,r) in sorted(zip(posteriors, all_meanings))[-5:]:
    #     print p, l, l.ori_relations, r, (r.distance, r.measurement.best_degree_class, r.measurement.best_distance_class ) if hasattr(r,'measurement') else 'No measurement'

    # meaning_posteriors = dict( zip(all_meanings,posteriors) )

    combined_heatmaps = []
    # big_heatmap2 = None
    for obj_lmk, meanings, heatmapss in loi_infos:

        big_heatmap1 = None
        for m,(h1,h2) in zip(meanings, heatmapss):
            lmk,rel = m
            p = posteriors[rel]*posteriors[lmk]
            if big_heatmap1 is None:
                big_heatmap1 = p*h1
                # big_heatmap2 = p*h2
            else:
                big_heatmap1 += p*h1
                # big_heatmap2 += p*h2

        # plt.figure()
        # plt.suptitle(sentence)
        # # plt.subplot(121)

        # probabilities1 = big_heatmap1.reshape( (len(xs),len(ys)) ).T
        # plt.pcolor(x, y, probabilities1, cmap='jet', edgecolors='none', alpha=0.7)
        # plt.colorbar()

        # for lmk in scene.landmarks.values():
        #     if isinstance(lmk.representation, GroupLineRepresentation):
        #         xx = [lmk.representation.line.start.x, lmk.representation.line.end.x]
        #         yy = [lmk.representation.line.start.y, lmk.representation.line.end.y]
        #         plt.fill(xx,yy,facecolor='none',linewidth=2)
        #     elif isinstance(lmk.representation, RectangleRepresentation):
        #         rect = lmk.representation.rect
        #         xx = [rect.min_point.x,rect.min_point.x,rect.max_point.x,rect.max_point.x]
        #         yy = [rect.min_point.y,rect.max_point.y,rect.max_point.y,rect.min_point.y]
        #         plt.fill(xx,yy,facecolor='none',linewidth=2)
        #         plt.text(rect.min_point.x+0.01,rect.max_point.y+0.02,lmk.name)

        # plt.plot(speaker.location.x,
        #          speaker.location.y,
        #          'bx',markeredgewidth=2)

        # plt.axis('scaled')
        # plt.axis([scene_bb.min_point.x, scene_bb.max_point.x, scene_bb.min_point.y, scene_bb.max_point.y])
        # plt.title('Likelihood of sentence given location(s)')

        # plt.subplot(122)

        # probabilities2 = big_heatmap2.reshape( (len(xs),len(ys)) ).T
        # plt.pcolor(x, y, probabilities2, cmap = 'jet', edgecolors='none', alpha=0.7)
        # plt.colorbar()

        # for lmk in scene.landmarks.values():
        #     if isinstance(lmk.representation, GroupLineRepresentation):
        #         xx = [lmk.representation.line.start.x, lmk.representation.line.end.x]
        #         yy = [lmk.representation.line.start.y, lmk.representation.line.end.y]
        #         plt.fill(xx,yy,facecolor='none',linewidth=2)
        #     elif isinstance(lmk.representation, RectangleRepresentation):
        #         rect = lmk.representation.rect
        #         xx = [rect.min_point.x,rect.min_point.x,rect.max_point.x,rect.max_point.x]
        #         yy = [rect.min_point.y,rect.max_point.y,rect.max_point.y,rect.min_point.y]
        #         plt.fill(xx,yy,facecolor='none',linewidth=2)
        #         plt.text(rect.min_point.x+0.01,rect.max_point.y+0.02,lmk.name)

        # plt.plot(speaker.location.x,
        #          speaker.location.y,
        #          'bx',markeredgewidth=2)

        # plt.axis('scaled')
        # plt.axis([scene_bb.min_point.x, scene_bb.max_point.x, scene_bb.min_point.y, scene_bb.max_point.y])
        # plt.title('Likelihood of location(s) given sentence')
        # plt.show()

        combined_heatmaps.append(big_heatmap1)

    return combined_heatmaps


def get_most_likely_object(scene, speaker, sentences):
    step = 0.04

    loi = [lmk for lmk in scene.landmarks.values() if lmk.name != 'table']
    all_heatmaps_dicts, xs, ys = speaker.generate_all_heatmaps(scene, step=step, loi=loi)

    loi_infos = []
    all_meanings = set()
    for obj_lmk,all_heatmaps_dict in zip(loi, all_heatmaps_dicts):
        all_heatmaps_tuples = []

        for lmk, d in all_heatmaps_dict.items():
            for rel, heatmaps in d.items():
                all_heatmaps_tuples.append( (lmk,rel,heatmaps) )

        # all_heatmaps_tuples = all_heatmaps_tuples[:10]

        lmks, rels, heatmapss = zip(*all_heatmaps_tuples)
        meanings = zip(lmks,rels)
        # print meanings
        all_meanings.update(meanings)
        loi_infos.append( (obj_lmk, meanings, heatmapss) )
    
    objects = []
    for sentence in sentences:
        lmk_probs = []
        try:
            combined_heatmaps = heatmaps_for_sentence(sentence, all_meanings, loi_infos, xs, ys, scene, speaker, step=step)
            
            for combined_heatmap,obj_lmk in zip(combined_heatmaps, loi):
                ps = [p for (x,y),p in zip(list(product(xs,ys)),combined_heatmap) if obj_lmk.representation.contains_point( Vec2(x,y) )]
                # print ps, xs.shape, ys.shape, combined_heatmap.shape
                lmk_probs.append( (sum(ps)/len(ps), obj_lmk) )
                
            top_p, top_lmk = sorted(lmk_probs, reverse=True)[0]
            lprobs, lmkss = zip(*lmk_probs)
            
            print
            print sorted(zip(np.array(lprobs)/sum(lprobs), [(l.name, l.color, l.object_class) for l in lmkss]), reverse=True)
            print 'I bet %f you are talking about a %s %s %s' % (top_p/sum(lprobs), top_lmk.name, top_lmk.color, top_lmk.object_class)
            objects.append(top_lmk)
        except Exception as e:
            print 'Unable to get object from sentence. ', e
            
    return objects


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sentence')
    parser.add_argument('-i', '--iterations', type=int, default=1)
    parser.add_argument('--return-object', action='store_true')
    parser.add_argument('-n', '--number_of_sentences', type=int, default=1)
    args = parser.parse_args()

    if not args.return_object:
        posteriors = get_sentence_posteriors(args.sentence, args.iterations)

        for m,p in sorted(posteriors, key=itemgetter(1)):
            print 'Meaning: %s \t\t Probability: %0.4f' % (m,p)
    else:
        scene, speaker = construct_training_scene()
        sentences = []
        for _ in range(args.number_of_sentences):
            sentences.append( raw_input('Location sentence: ') )
        get_most_likely_object(scene, speaker, sentences)

