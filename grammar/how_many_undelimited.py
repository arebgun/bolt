# coding: utf-8
import sys
sys.path.insert(1,'..')
import semantics as sem

import automain
import utils
# import random
import common as cmn
import language_user as lu
import lexical_items as li
import constructions as st
import construction as cs
import constraint
import shelve
import constructions as cs
import IPython
f = shelve.open('turk_extrinsic_training3.shelf')
f.keys()
f['training_data'][0]

parses = zip(*zip(*zip(*zip(*f['training_data'])[-1])[0])[-1])[0]
IPython.embed()
# count = 0
# for parse in parses:
#     if isinstance(parse.constituents[1],cs.ExtrinsicReferringExpression) and not isinstance(parse.constituents[1].constituents[1].constituents[1].constituents[0],cs.PartOfRelation):
#         count += 1
        
# count
# len(parses)