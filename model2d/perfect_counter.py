#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import sys
sys.path.append("..")
import csv

from models import (Location, Word, Production, Bigram, Trigram,
                    CWord, CProduction, SentenceParse, session)
import utils
from utils import parent_landmark, count_lmk_phrases, get_meaning, rel_type, lmk_id, get_lmk_ori_rels_str, scene
from parse import get_modparse
from semantics.landmark import Landmark
from semantics.representation import PointRepresentation
from planar import Vec2
from nltk.tree import ParentedTree
from myrandom import random
random = random.random

from sqlalchemy import func
from sqlalchemy.orm import aliased

from semantics.run import construct_training_scene



def save_tree(tree, loc, rel, lmk, parent=None):
    if len(tree.productions()) == 1:
        # if this tree only has one production
        # it means that its child is a terminal (word)
        word = Word()
        word.word = tree[0]
        word.pos = tree.node
        word.parent = parent
        word.location = loc
    else:
        prod = Production()
        prod.lhs = tree.node
        prod.rhs = ' '.join(n.node for n in tree)
        prod.parent = parent
        prod.location = loc

        # some productions are related to semantic representation
        if prod.lhs == 'RELATION':
            prod.relation = rel_type(rel)
            if hasattr(rel, 'measurement'):
                prod.relation_distance_class = rel.measurement.best_distance_class
                prod.relation_degree_class = rel.measurement.best_degree_class

        elif prod.lhs == 'LANDMARK-PHRASE':
            prod.landmark = lmk_id(lmk)
            prod.landmark_class = lmk.object_class
            prod.landmark_orientation_relations = get_lmk_ori_rels_str(lmk)
            prod.landmark_color = lmk.color
            # next landmark phrase will need the parent landmark
            lmk = parent_landmark(lmk)

        elif prod.lhs == 'LANDMARK':
            # LANDMARK has the same landmark as its parent LANDMARK-PHRASE
            prod.landmark = parent.landmark
            prod.landmark_class = parent.landmark_class
            prod.landmark_orientation_relations = parent.landmark_orientation_relations
            prod.landmark_color = parent.landmark_color

        # save subtrees, keeping track of parent
        for subtree in tree:
            save_tree(subtree, loc, rel, lmk, prod)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-i', '--iterations', type=int, default=1)
    args = parser.parse_args()

    unique_sentences = {}

    # scene, speaker = scene.scene, scene.speaker

    for i in range(args.iterations):
        print 'sentence', i

        if (i % 50) == 0:
            scene, speaker = construct_training_scene(True)
            utils.scene.set_scene(scene,speaker)
            table = scene.landmarks['table'].representation.rect
        t_min = table.min_point
        t_max = table.max_point
        t_w = table.width
        t_h = table.height

        xloc,yloc = random()*t_w+t_min.x, random()*t_h+t_min.y
        trajector = Landmark( 'point', PointRepresentation(Vec2(xloc,yloc)), None, Landmark.POINT)
        sentence, rel, lmk = speaker.describe(trajector, scene, False, 1)
        parsestring, modparsestring = get_modparse(sentence)
        unique_sentences[sentence] = (parsestring, modparsestring)

        # convert variables to the right types
        loc = (xloc, yloc)
        parse = ParentedTree.parse(parsestring)
        modparse = ParentedTree.parse(modparsestring)

        # how many ancestors should the sampled landmark have?
        num_ancestors = count_lmk_phrases(modparse) - 1

        if num_ancestors == -1:
            print 'Failed to parse %d [%s] [%s] [%s]' % (i, sentence, parse, modparse)
            continue

        assert(not isinstance(lmk, tuple))
        assert(not isinstance(rel, tuple))

        if args.verbose:
            print 'utterance:', repr(sentence)
            print 'location: %s' % repr(loc)
            print 'landmark: %s (%s)' % (lmk, lmk_id(lmk))
            print 'relation: %s' % rel_type(rel)
            print 'parse:'
            print parse.pprint()
            print 'modparse:'
            print modparse.pprint()
            print '-' * 70

        location = Location(x=xloc, y=yloc)
        save_tree(modparse, location, rel, lmk)
        Bigram.make_bigrams(location.words)
        Trigram.make_trigrams(location.words)

        if i % 200 == 0: session.commit()

    if SentenceParse.query().count() == 0:
        for sentence,(parse,modparse) in unique_sentences.items():
            SentenceParse.add_sentence_parse_blind(sentence, parse, modparse)
    else:
        for sentence,(parse,modparse) in unique_sentences.items():
            SentenceParse.add_sentence_parse(sentence, parse, modparse)

    session.commit()

    print 'counting ...'

    # count words
    w1 = aliased(Word)
    w2 = aliased(Word)
    parent = aliased(Production)
    qry = session.query(w1.word, w2.word, w2.pos,
                        parent.lhs, parent.landmark, parent.landmark_class,
                        parent.landmark_orientation_relations, parent.landmark_color,
                        parent.relation, parent.relation_distance_class,
                        parent.relation_degree_class, func.count(w2.id)) \
                                .outerjoin(w1,Bigram.w1) \
                                .join(w2,Bigram.w2) \
                                .join(parent,w2.parent) \
                                .group_by(w1.word, w2.word, w2.pos, parent.lhs,
                                          parent.landmark, parent.landmark_class,
                                          parent.landmark_orientation_relations,
                                          parent.landmark_color, parent.relation,
                                          parent.relation_distance_class,
                                          parent.relation_degree_class)
    for row in qry:
        cw = CWord(word=row[1],
                   prev_word=row[0],
                   pos=row[2],
                   landmark=row[4],
                   landmark_class=row[5],
                   landmark_orientation_relations=row[6],
                   landmark_color=row[7],
                   relation=row[8],
                   relation_distance_class=row[9],
                   relation_degree_class=row[10],
                   count=row[11])

    # count productions with no parent
    parent = aliased(Production)
    qry = session.query(Production.lhs, Production.rhs,
                        Production.landmark, Production.landmark_class, Production.landmark_orientation_relations, Production.landmark_color,
                        Production.relation, Production.relation_distance_class,
                        Production.relation_degree_class, func.count(Production.id)).\
                  filter_by(parent=None).\
                  group_by(Production.lhs, Production.rhs,
                           Production.landmark, Production.landmark_class, Production.landmark_orientation_relations, Production.landmark_color,
                           Production.relation, Production.relation_distance_class,
                           Production.relation_degree_class)
    for row in qry:
        cp = CProduction(lhs=row[0],
                         rhs=row[1],
                         landmark=row[2],
                         landmark_class=row[3],
                         landmark_orientation_relations=row[4],
                         landmark_color=row[5],
                         relation=row[6],
                         relation_distance_class=row[7],
                         relation_degree_class=row[8],
                         count=row[9])

    # count productions with parent
    parent = aliased(Production)
    qry = session.query(Production.lhs, Production.rhs,
                        parent.lhs, Production.landmark, Production.landmark_class, Production.landmark_orientation_relations, Production.landmark_color,
                        Production.relation, Production.relation_distance_class,
                        Production.relation_degree_class, func.count(Production.id)).\
                  join(parent, Production.parent).\
                  group_by(Production.lhs, Production.rhs,
                           parent.lhs, Production.landmark, Production.landmark_class, Production.landmark_orientation_relations, Production.landmark_color,
                           Production.relation, Production.relation_distance_class,
                           Production.relation_degree_class)
    for row in qry:
        cp = CProduction(lhs=row[0],
                         rhs=row[1],
                         parent=row[2],
                         landmark=row[3],
                         landmark_class=row[4],
                         landmark_orientation_relations=row[5],
                         landmark_color=row[6],
                         relation=row[7],
                         relation_distance_class=row[8],
                         relation_degree_class=row[9],
                         count=row[10])

    session.commit()
