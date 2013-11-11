import random
import numpy as np
import operator as op
import utils
import lexical_items as li
import constructions as structs
import partial_parser as pp
from all_parse_generator import AllParseGenerator as apg

class LanguageUser(object):

    def __init__(self, lexicon, constructicon):
        self.lexicon = lexicon
        self.constructicon = constructicon

    def set_context(self, context):
        self.context = context
        self.complete_The_object__parses()

    def choose_referent(self, referring_expression, return_all_tied=False):
        parses = self.parse(utterance=referring_expression)
        parses = sorted(parses, key=op.attrgetter('hole_width'))
        # for parse in parses:
        #     utils.logger(parse.current)
        #     utils.logger(parse.hole_width)
        #     raw_input()
        parse = parses[0].current[0]
        potential_referents = self.context.get_potential_referents()
        apps = parse.sempole().ref_applicabilities(self.context, 
                                                   potential_referents)
        app_items = sorted(apps.items(), key=op.itemgetter(1), reverse=True)
        best_app = app_items[0][1]
        best_refs = [ref for ref, app in app_items if app == best_app]
        if return_all_tied:
            return best_refs
        else:
            return random.choice(best_refs)

    def parse(self, utterance):
        lexical_parses = pp.get_all_lexical_parses(sentence=utterance, 
                                                   lexicon=self.lexicon)
        parses=pp.get_all_construction_parses(lexical_parses=lexical_parses,
                                              constructicon=self.constructicon,
                                              max_holes=1)
        return parses

    def generate(self, referent):
        raise NotImplementedError

    def complete_The_object__parses(self):
        target = structs.RelationNounPhrase
        pattern_start = [[structs.NounPhrase([li.objct])]]
        current_depths = {structs.ReferringExpression:1}
        rnpparses = apg.finish_parse(targetclass=target,
                                     pattern_start=pattern_start,
                                     lexicon=self.lexicon,
                                     constructicon=self.constructicon,
                                     current_depths=current_depths)
        parses = [structs.ExtrinsicReferringExpression([li.the, parse]) 
                  for parse in rnpparses]
        self.The_object__parses = parses

    def weight_parses(self, referent, parses):
        potential_referents = self.context.get_potential_referents()
        scores = []
        for parse in parses:
            applicabilities = parse.sempole().ref_applicabilities(self.context, 
                                    potential_referents)
            applicability = float(applicabilities[referent])
            sumav = sum(applicabilities.values())
            if sumav == 0: old_score = 0
            else: old_score = (applicability**2) / sumav
            scores.append(old_score)

        result = sorted(zip(scores, parses), reverse=True)
        return result

    @staticmethod
    def choose_top_parse(sorted_parses):
        return sorted_parses[0][1]

    @staticmethod
    def sample_parse(sorted_parses):
        raise NotImplementedError
