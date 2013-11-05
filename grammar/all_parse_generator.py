#!/usr/bin/python

import sys
sys.path.insert(1,'..')
import pprint
import IPython
# import random as rand
import argparse
import automain
import numpy as np
import itertools as it
import collections as coll
import matplotlib.pyplot as plt
import common as cmn
import lexical_items as li
import construction as struct
import constructions as structs

import semantics as sem
import utils
import gen2_features as feats
import sempoles
import constraint as const

pp = pprint.PrettyPrinter()

class AllParseGenerator(object):
    
    @classmethod
    def generate_parses(cls, targetclass, 
                        max_depths={structs.ReferringExpression:2}, 
                        current_depths=coll.defaultdict(float)):

        if targetclass in max_depths:
            current_depths[targetclass]+=1
            if current_depths[targetclass] > max_depths[targetclass]:
                return []

        if issubclass(targetclass, struct.LexicalItem):
            completed_matches = [item for item in li.lexical_items_list
                                 if isinstance(item, targetclass)]

            return completed_matches
        #else:

        matches = [construction for construction in structs.constructions_list
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

                # if targetclass == structs.ReferringExpression:
                #     utils.logger('%s %s %i' %(matching_const, subtargetclass, len(sub_matches)))

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

    @classmethod
    def finish_parse(cls, targetclass, pattern_start,
                     max_depths={structs.ReferringExpression:2}, 
                     current_depths=coll.defaultdict(float)):
        if targetclass in max_depths:
            current_depths[targetclass]+=1
            if current_depths[targetclass] > max_depths[targetclass]:
                return []

        completed_matches = []

        pattern_matches = pattern_start
        for subtargetclass in targetclass.pattern[len(pattern_start):]:
            utils.logger('Matching %s' % subtargetclass)
            sub_matches = cls.generate_parses(subtargetclass, 
                                              max_depths,
                                              current_depths)
            if sub_matches is []:
                break
            else:
                pattern_matches.append(sub_matches)
            # utils.logger(pattern_matches)

        utils.logger(len(pattern_matches[1]))

        if len(pattern_matches) < len(targetclass.pattern):
            # Not all parts of pattern had matches
            return []
        else:
            # order = matching_const.application_order
            constituent_tuples = list(it.product(*pattern_matches))
            for constituent_tuple in constituent_tuples:
                try:
                    completed_match = targetclass(constituent_tuple)
                except AssertionError as ass:
                    utils.logger(ass)
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
        potential_referent_scores = cmn.Applicabilities(pairs)
        return potential_referent_scores

@automain.automain
def test():

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s', '--scene_directory', type=str)
    args = argparser.parse_args()

    scene_descs = sem.run.read_scenes(args.scene_directory,
                                  normalize=True,image=True)
    print

    scene, speaker, image = scene_descs[2]
    # plt.ion()
    plt.imshow(image)
    trajector = scene.landmarks.values()[0]
    print trajector

    context = Context(scene, speaker)
    potential_referents = context.get_potential_referents()
    # for (pot_ref,) in potential_referents:
    #     if str(pot_ref) == 'object_2 ORANGE CYLINDER':
    #         the_oc = pot_ref
    #     if str(pot_ref) == 'object_1 RED SPHERE':
    #         the_rs = pot_ref

    # angle1 = feats.angle_between.observe(the_rs, the_oc, speaker.location)
    # angle2 = feats.angle_between.observe(the_rs, the_oc, speaker.get_head_on_viewpoint(the_oc))
    # print angle1, sempoles.right_func(angle1)
    # print angle2, sempoles.right_func(angle2)

    # the_orange_cylinder = structs.ReferringExpression([
    #                         li.the,
    #                         structs.AdjectiveNounPhrase([
    #                             structs.AdjectivePhrase([
    #                                 li.orange
    #                                 ]),
    #                             li.cylinder
    #                             ])
    #                         ])

    # lmk_apps = the_orange_cylinder.sempole().ref_applicabilities(context, potential_referents)
    # print 'lmk_apps', lmk_apps

    # ttro = const.RelationConstraint(feature=feats.angle_between,
    #                                 prob_func=sempoles.right_func)
    # viewpoint = context.speaker.location
    # print ttro.entity_applicability(context, the_rs, lmk_apps, viewpoint=viewpoint)

    # relata_apps = lmk_apps
    # entity = the_rs
    # ps = []
    # for relatum, relatum_app in relata_apps.items():
    #     print 'relatum', relatum
    #     p = 1.0
    #     for relentity in relatum:
    #         print '  relentity', relentity
    #         observation = ttro.feature.observe(entity, 
    #                                             relentity,
    #                                             # context.speaker.get_head_on_viewpoint(relentity),
    #                                             viewpoint=viewpoint)
    #         print '  observation', observation
    #         probability = ttro.probability_func(observation)
    #         print '  probability', probability
    #         p*=probability
    #     print 'p', p
    #     # entity_app += p*relatum_app/relatum_app_sum
    #     ps.append(p*relatum_app)
    # ps = np.nan_to_num(ps)
    # ps = ps**2/ps.sum()
    # print 'ps', ps
    # print 'max(ps)', max(ps)

    # exit()

    # to_the_right_of = structs.OrientationRelation([
    #                     li.to,
    #                     li.the,
    #                     li.right,
    #                     li.of
    #                     ])

    # ttrotoc = structs.RelationLandmarkPhrase([
    #             to_the_right_of,
    #             the_orange_cylinder
    #             ])

    # ref_apps=ttrotoc.sempole().ref_applicabilities(context, potential_referents)
    # print 'ref_apps',ref_apps


    # exit()

    # target = structs.ReferringExpression
    # parses = AllParseGenerator.generate_parses(targetclass=target,
    #             max_depths={structs.ReferringExpression:2})


    current_depths = coll.defaultdict(int)
    current_depths[structs.ReferringExpression] = 1
    # target = structs.ExtrinsicReferringExpression
    # parses = AllParseGenerator.generate_parses(targetclass=target,
    #             max_depths={structs.ReferringExpression:2},
    #             current_depths=current_depths)

    target = structs.RelationNounPhrase
    pattern_start = [[structs.NounPhrase([li.objct])]]
    parses = AllParseGenerator.finish_parse(targetclass=target,
                                            pattern_start=pattern_start,
                                            current_depths=current_depths)
    parses = [structs.ExtrinsicReferringExpression([li.the, parse]) 
              for parse in parses]

    print 'parses generated:', len(parses)
    print

    # for parse in parses:
    #     print parse.prettyprint()
    #     raw_input()

    # exit()
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
