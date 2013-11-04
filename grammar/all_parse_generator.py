#!/usr/bin/python

import sys
sys.path.insert(1,'..')
import pprint
import IPython
# import random as rand
import numpy as np
import itertools as it
import collections as coll
from automain import automain
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from constructions import constructions_list, ReferringExpression
from construction import LexicalItem
from lexical_items import lexical_items_list
from semantics import run
from common import Applicabilities

pp = pprint.PrettyPrinter()

class AllParseGenerator(object):
    
    @classmethod
    def generate_parses(cls, targetclass, 
                        max_depths={ReferringExpression:2}, 
                        current_depths=coll.defaultdict(float)):

        if targetclass in max_depths:
            current_depths[targetclass]+=1
            if current_depths[targetclass] > max_depths[targetclass]:
                return []

        if issubclass(targetclass, LexicalItem):
            completed_matches = [item for item in lexical_items_list
                                 if isinstance(item, targetclass)]

            return completed_matches
        #else:

        matches = [construction for construction in constructions_list
                   if issubclass(construction, targetclass)]

        # for each matching construction, find all the ways to fulfill pattern
        completed_matches = []
        for matching_const in matches:

                
            pattern_matches = []
            for subtargetclass in matching_const.pattern:
                sub_matches = cls.generate_parses(subtargetclass, 
                                                  max_depths,
                                                  current_depths)
                if sub_matches is []:
                    break
                else:
                    pattern_matches.append(sub_matches)

            if len(pattern_matches) < len(matching_const.pattern):
                # Not all parts of pattern had matches
                pass# TODO
            else:
                # order = matching_const.application_order
                constituent_tuples = list(it.product(*pattern_matches))
                for constituent_tuple in constituent_tuples:
                    try:
                        completed_match = matching_const(constituent_tuple)
                    except Exception as e:
                        print e
                    else:
                        completed_matches.append(completed_match)

        return completed_matches

class Context(object):

    def __init__(self, scene, speaker):
        self.scene = scene
        self.speaker = speaker
        self.entities = scene.landmarks.values()
        self.potential_referents = [(entity,) for entity in self.entities]
        # for r in range(2, min(len(self.entities),2)+1):
        #     self.potential_referents.extend(it.permutations(self.entities,r))

    def get_entities(self):
        return list(self.entities)

    def get_potential_referents(self):
        return list(self.potential_referents)

    def get_potential_referent_scores(self):
        pairs = zip(self.potential_referents,[1]*len(self.potential_referents))
        potential_referent_scores = Applicabilities(pairs)
        return potential_referent_scores

@automain
def test():

    parser = ArgumentParser()
    parser.add_argument('-s', '--scene_directory', type=str)
    args = parser.parse_args()

    scene_descs = run.read_scenes(args.scene_directory,
                                  normalize=True,image=True)
    print

    target = ReferringExpression
    parses = AllParseGenerator.generate_parses(targetclass=target,
                max_depths={ReferringExpression:2})

    print 'parses generated:', len(parses)
    print


    scene, speaker, image = scene_descs[2]
    # plt.ion()
    plt.imshow(image)
    trajector = scene.landmarks.values()[0]
    print trajector

    context = Context(scene, speaker)
    potential_referents = context.get_potential_referents()

    # for parse in parses:
    #     print parse.prettyprint()
    #     print parse.sempole()
    #     print
    #     pp.pprint(parse.sempole().applicabilities(context))
    #     print
    #     print

    scores = []
    for parse in parses:
        # print parse.prettyprint()
        # print parse.sempole()
        # print
        # print
        applicabilities = parse.sempole().ref_applicabilities(context, 
                                                        potential_referents)
        applicability = float(applicabilities[(trajector,)])
        old_score = (applicability**2) / sum(applicabilities.values())
        if np.isnan(old_score): old_score = 0
        scores.append(old_score)
        # applicability_share = applicability / sum(applicabilities.values())
        # del applicabilities[(trajector,)]
        # print 'Trajector:', trajector
        # print
        # print parse.prettyprint()
        # print 'applicatility', applicability
        # print 'app share', applicability_share
        # print 'old score', old_score
        # raw_input()

    print 'Trajector:', trajector
    print
    for score, parse in sorted(zip(scores, parses), reverse=True)[:20]:
        print '-'*10
        print parse.prettyprint()
        print parse.print_sentence()
        print 'Score:', score
    plt.show()
