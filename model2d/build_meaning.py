#!/usr/bin/env python
# coding: utf-8

from __future__ import division

from operator import itemgetter

import numpy as np
from nltk.tree import ParentedTree
from parse import get_modparse

from utils import (
    parent_landmark,
    get_meaning,
    rel_type,
    m2s,
    count_lmk_phrases,
    logger,
    get_lmk_ori_rels_str,
    entropy_of_counts,
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
from collections import Counter

from string import rjust

def get_total_entropy():
    cp_db = CProduction.query
    cpcolumns = list(CProduction.__table__.columns)[3:]
    totalss = get_query_totalss(cp_db,cpcolumns)
    cptotalentropy = {}
    for name,totals in zip(cpcolumns[:-1],totalss):
        ent = entropy_of_counts( totals.values() )
        cptotalentropy[name.name] = ent

    cw_db = CWord.query
    cwcolumns = list(CWord.__table__.columns)[3:]
    totalss = get_query_totalss(cw_db,cwcolumns)
    cwtotalentropy = {}
    for name,totals in zip(cwcolumns[:-1],totalss):
        ent = entropy_of_counts( totals.values() )
        cwtotalentropy[name.name] = ent

    printlength = max( [len(column.name) for column in \
                        list(CProduction.__table__.columns)[3:-1] + \
                        list(CWord.__table__.columns)[3:-1] ] )

    return cptotalentropy,cpcolumns,cwtotalentropy,cwcolumns,printlength

def get_query_totalss(q,columns):
    all_columns = zip(  *list( q.values( *columns ) )  )
    counts = all_columns[-1]
    all_columns = all_columns[:-1]
    
    totalss = []
    for column in all_columns:
        cs = Counter()
        for entry,count in zip(column,counts):
            cs[entry] += count
        totalss.append(cs)
    return totalss

def print_totalss_entropy(totalss,totalentropy,columns,printlength):
    print rjust('column',printlength), rjust('context',7), rjust('overall',7), 'best'
    print rjust('',printlength), rjust('entropy',7), rjust('entropy',7)
    for name,totals in zip(columns[:-1],totalss):
        print rjust(name.name,printlength), \
              rjust("%02.4f" % entropy_of_counts( totals.values() ),7), \
              rjust("%02.4f" % totalentropy[name.name],7), \
              zip(*sorted(zip(*reversed(zip(*totals.items()))),reverse=True))[1]
    print
    print

def print_tree_entropy(tree, 
                       cptotalentropy=None,
                       cpcolumns=None,
                       cwtotalentropy=None,
                       cwcolumns=None,
                       printlength=None):

    if cptotalentropy is None:
        cptotalentropy,cpcolumns,cwtotalentropy,cwcolumns,printlength = get_total_entropy()

    lhs = tree.node
    if isinstance(tree[0], ParentedTree): rhs = ' '.join(n.node for n in tree)
    else: rhs = ' '.join(n for n in tree)

    print tree
    print '+++',lhs,'-->',rhs,'+++'

    if lhs in NONTERMINALS:
        cp_db = CProduction.get_production_counts(lhs=lhs,rhs=rhs)
        totalss = get_query_totalss(cp_db,cpcolumns)
        print_totalss_entropy(totalss,cptotalentropy,cpcolumns,printlength)

        for subtree in tree:
            print_tree_entropy(subtree, 
                               cptotalentropy, 
                               cpcolumns,
                               cwtotalentropy, 
                               cwcolumns,
                               printlength)
    else:
        cw_db = CWord.get_word_counts(pos=lhs,word=rhs)
        totalss = get_query_totalss(cw_db,cwcolumns)
        print_totalss_entropy(totalss,cwtotalentropy,cwcolumns,printlength)


def build_meaning(tree,
                  parent=None,
                  parts=[],
                  cptotalentropy=None,
                  cpcolumns=None,
                  cwtotalentropy=None,
                  cwcolumns=None,
                  threshold=0.75):
    
    cptotalentropy,cpcolumns,cwtotalentropy,cwcolumns,printlength = get_total_entropy()

    lhs = tree.node
    if isinstance(tree[0], ParentedTree): rhs = ' '.join(n.node for n in tree)
    else: rhs = ' '.join(n for n in tree)

    print '+++',lhs,'-->',rhs,'+++'
    if lhs in NONTERMINALS:

        if not lhs == 'LOCATION-PHRASE':

            if lhs == 'RELATION':
                parts.append( ('relation',Counter()) )
            elif lhs == parent == 'LANDMARK-PHRASE':
                parts.append( ('parent-landmark',Counter()) )
            elif lhs == 'LANDMARK-PHRASE':
                parts.append( ('landmark',Counter()) )

            cp_db = CProduction.get_production_counts(lhs=lhs,rhs=rhs)
            totalss = get_query_totalss(cp_db,cpcolumns)

            for name,totals in zip(cpcolumns[:-1],totalss):
                ent = entropy_of_counts( totals.values() )
                totent = cptotalentropy[name.name]
                if ent < threshold*totent:
                    parts[-1][1][ "%s = %s" % (name.name, max(zip(*reversed(zip(*totals.items()))))[1]) ]+=1


        for subtree in tree:
            parts = build_meaning(subtree,
                                  lhs,
                                  parts,
                                  cptotalentropy, 
                                  cpcolumns,
                                  cwtotalentropy, 
                                  cwcolumns,
                                  threshold)
    else:

        cw_db = CWord.get_word_counts(pos=lhs,word=rhs)
        totalss = get_query_totalss(cw_db,cwcolumns)

        for name,totals in zip(cwcolumns[:-1],totalss):
            ent = entropy_of_counts( totals.values() )
            totent = cwtotalentropy[name.name]
            if ent < threshold*totent:
                parts[-1][1][ "%s = %s" % (name.name, max(zip(*reversed(zip(*totals.items()))))[1]) ]+=1

    return parts




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('-n', '--num_iterations', type=int, default=1)
    # parser.add_argument('-l', '--location', type=Point)
    # parser.add_argument('--consistent', action='store_true')
    parser.add_argument('sentence')
    args = parser.parse_args()

    scene, speaker = construct_training_scene()

    print 'parsing ...'
    modparse = get_modparse(args.sentence)
    t = ParentedTree.parse(modparse)
    print '\n%s\n' % t.pprint()

    print_tree_entropy(t)

    raw_input()

    parts = build_meaning(t)
    for part in parts:
        print "Suggested for", part[0]
        items = sorted(part[1].items(), key=lambda x: x[1],reverse=True)
        for item in items:
            print '    ',rjust(item[0],40),item[1]
        print