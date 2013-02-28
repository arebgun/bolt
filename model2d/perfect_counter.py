#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import sys
import argparse
sys.path.append("..")

import models
from models import Location, Production, CProduction, SentenceParse, session

import utils
from utils import get_landmark_parent_chain, rel_type, lmk_id, get_lmk_ori_rels_str, logger

import parse_adios
from semantics.run import construct_training_scene
from semantics.landmark import Landmark
from semantics.representation import PointRepresentation

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

    for i in range(args.num_sentences):
        if (i % args.num_per_scene) == 0:
            if i != 0: sys.stdout.write('\bDone.\n')
            sys.stdout.write('Generating sentences for scene %d:  ' % (int(i/args.num_per_scene) + 1))
            num_objects = int(random() * (args.max_objects - args.min_objects) + args.min_objects)
            scene, speaker = construct_training_scene(random=True, num_objects=num_objects)
            utils.scene.set_scene(scene, speaker)
            table = scene.landmarks['table'].representation.rect

        sys.stdout.write("\b%s" % syms[i % len(syms)])
        sys.stdout.flush()

        t_min = table.min_point
        t_max = table.max_point
        t_w = table.width
        t_h = table.height

        # generate a random location and generate a sentence describing it
        xloc, yloc = random() * t_w + t_min.x, random() * t_h + t_min.y
        trajector = Landmark('point', PointRepresentation( Vec2(xloc,yloc) ), None, Landmark.POINT)
        sentence, rel, lmk = speaker.describe(trajector, scene, False, 1)

        try:
            parsestring = parse_adios.parse(sentence)
        except Exception as pe:
            sys.stdout.write('\b\n')
            sys.stdout.flush()
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

    sys.stdout.write('\bDone.\n')

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
