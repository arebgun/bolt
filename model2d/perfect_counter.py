#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import sys
import argparse
sys.path.append("..")

import models
from models import Location, Production, CProduction, SentenceParse, session

import utils
from utils import get_landmark_parent_chain
from utils import rel_type
from utils import lmk_id
from utils import get_lmk_ori_rels_str
from utils import logger
from utils import ngrams

from parse_adios import parse as parse_adios
from parse_bllip import parse as parse_bllip
from semantics.run import construct_training_scene
from semantics.landmark import Landmark
from semantics.representation import PointRepresentation
from semantics.speaker import Speaker

from myrandom import random
random = random.random

from planar import Vec2
from nltk.tree import ParentedTree
from sqlalchemy import func
from sqlalchemy.orm import aliased


def save_tree(tree, loc, rel, lmks, parent=None):
    if isinstance(tree, ParentedTree):
        for lmk in lmks:
            prod = Production()
            prod.lhs = tree.node
            prod.rhs = ' '.join(n.node if isinstance(n, ParentedTree) else n for n in tree)
            prod.parent = parent
            prod.location = loc

            prod.relation = rel_type(rel)
            if hasattr(rel, 'measurement'):
                prod.relation_distance_class = rel.measurement.best_distance_class
                prod.relation_degree_class = rel.measurement.best_degree_class

            prod.landmark = lmk_id(lmk)
            prod.landmark_class = lmk.object_class
            prod.landmark_orientation_relations = get_lmk_ori_rels_str(lmk)
            prod.landmark_color = lmk.color

            if prod.lhs == 'S':
                tokens = prod.rhs.split()
                ngram_list = ngrams(tokens)

                for ngram in ngram_list:
                    s_ngram = Production()
                    s_ngram.lhs = prod.lhs
                    s_ngram.rhs = ' '.join(ngram)
                    s_ngram.parent = prod.parent
                    s_ngram.location = prod.location
                    s_ngram.relation_distance_class = prod.relation_distance_class
                    s_ngram.relation_degree_class = prod.relation_degree_class
                    s_ngram.landmark = prod.landmark
                    s_ngram.landmark_class = prod.landmark_class
                    s_ngram.landmark_orientation_relations = prod.landmark_orientation_relations
                    s_ngram.landmark_color = prod.landmark_color

            # save subtrees, keeping track of parent
            for subtree in tree:
                save_tree(subtree, loc, rel, lmks, prod)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-n', '--num-sentences', type=int, default=1)
    parser.add_argument('-i', '--num-per-scene', type=int, default=50)
    parser.add_argument('--min_objects', type=int, default=1)
    parser.add_argument('--max-objects', type=int, default=7)
    args = parser.parse_args()

    models.engine.echo = False
    models.create_all()

    unique_sentences = {}
    num_failed = 0
    syms = ['\\', '|', '/', '-']
    console_write = sys.stdout.write
    console_flush = sys.stdout.flush
    describe_trajector = Speaker.describe

    for i in range(args.num_sentences):
        if (i % args.num_per_scene) == 0:
            if i != 0: console_write('\bDone.\n')
            console_write('Generating sentences for scene %d:  ' % (int(i/args.num_per_scene) + 1))
            num_objects = int(random() * (args.max_objects - args.min_objects) + args.min_objects)
            scene, speaker = construct_training_scene(random=True, num_objects=num_objects)
            utils.scene.set_scene(scene, speaker)
            table = scene.landmarks['table'].representation.rect
            t_min = table.min_point
            t_max = table.max_point
            t_w = table.width
            t_h = table.height

        console_write("\b%s" % syms[i % len(syms)])
        console_flush()

        # generate a random location and generate a sentence describing it
        xloc, yloc = random() * t_w + t_min.x, random() * t_h + t_min.y
        trajector = Landmark('point', PointRepresentation( Vec2(xloc, yloc) ), None, Landmark.POINT)
        sentence, rel, lmk = describe_trajector(speaker, trajector, scene, False, 1)

        try:
            # parsestring = parse_adios(sentence)
            parsestring = parse_bllip(sentence)
        except Exception as pe:
            console_write('\b\n')
            console_flush()
            logger('Failed to parse sentence "%s"' % sentence, 'warning')
            logger(str(pe), 'warning')
            num_failed += 1
            continue

        unique_sentences[sentence] = parsestring

        # convert variables to the right types
        loc = (xloc, yloc)
        parse = ParentedTree.parse(parsestring)

        assert(not isinstance(lmk, tuple))
        assert(not isinstance(rel, tuple))

        if args.verbose:
            logger( '%s; (%s, %s)' % (repr(loc), lmk, rel_type(rel)) )
            logger( '%s; %s' % (repr(sentence), parse.pprint) )

        location = Location(x=xloc, y=yloc)
        save_tree( parse, location, rel, get_landmark_parent_chain(lmk) )

        if i % 200 == 0: session.commit()

    console_write('\bDone.\n')

    if SentenceParse.query().count() == 0:
        for sentence, parse in unique_sentences.items():
            SentenceParse.add_sentence_parse_blind(sentence, parse, parse)
    else:
        for sentence, parse in unique_sentences.items():
            SentenceParse.add_sentence_parse(sentence, parse, parse)

    session.commit()

    logger('Failed to parse %d out of %d (%.2f%%) sentences' % (num_failed, args.num_sentences, num_failed/args.num_sentences*100))
    logger('Counting ...')

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
