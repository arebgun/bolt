import random
import numpy as np
import operator as op
import utils
import lexical_items as li
import constructions as st
import partial_parser as pp
from all_parse_generator import AllParseGenerator as apg

import os
import sqlalchemy as alc
import gen2_features as g2f
import probability_function as pf
import domain as dom

class LanguageUser(object):
    db_suffix = '_memories.db'

    def __init__(self, name, lexicon, structicon, 
                       remember=True, reset=False):
        self.name = name
        self.db_name = self.name + self.db_suffix
        self.lexicon = lexicon
        self.structicon = structicon
        self.remember = remember

        db_name = self.name + self.db_suffix
        if reset and os.path.isfile(db_name):
            os.remove(self.name + self.db_suffix)
            self.create_memorybase()

    def create_memorybase(self):
        engine = alc.create_engine('sqlite:///'+self.db_name, echo=False)
        metadata = alc.MetaData()
        alc.Table('scenes', metadata,
            alc.Column('id', alc.Integer, primary_key=True))

        feature_columns = []
        for feature in g2f.feature_list:
            type_map = {bool:alc.Boolean, int:alc.Integer, 
                        float:alc.Float, str:alc.String}
            feature_columns.append(alc.Column(feature.domain.name, 
                                            type_map[feature.domain.datatype]))

        alc.Table('unknown_structs', metadata,
            alc.Column('id', alc.Integer, primary_key=True),
            alc.Column('scene_id', None, alc.ForeignKey('scenes.id')),
            alc.Column('construction', alc.String),
            alc.Column('string', alc.String),
            alc.Column('exemplary', alc.Boolean),
            alc.Column('lmk_prob', alc.Float),
            *feature_columns)
        metadata.create_all(engine)


    def connect_to_memories(self):
        utils.logger('Connecting to memories')
        engine = alc.create_engine('sqlite:///'+self.db_name, echo=False)
        meta = alc.MetaData()
        meta.reflect(bind=engine)
        self.scenes = meta.tables['scenes']
        self.unknown_structs = meta.tables['unknown_structs']
        self.connection = engine.connect()


    def copy(self):
        return LanguageUser(self.name, self.lexicon, self.structicon, 
                            remember=self.remember, reset=False)

    def set_context(self, context):
        utils.logger(self.name)
        if self.remember:
            self.scene_key = self.connection.execute(self.scenes.insert()
                                ).inserted_primary_key[0]
        self.context = context
        self.complete_The_object__parses()
        self.generate_landmark_parses()

    def choose_referent(self, referring_expression, return_all_tied=False):
        parses = self.parse(utterance=referring_expression)
        parses = sorted(parses, key=op.attrgetter('hole_width'))
        # for parse in parses:
        #     utils.logger(parse.current)
        #     utils.logger(parse.hole_width)
        #     raw_input()
        parse = parses[0].current[0]
        potential_referents = self.context.get_all_potential_referents()
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
                                              structicon=self.structicon,
                                              max_holes=1)
        return parses

    def generate_landmark_parses(self):
        arts = [lex for lex in self.lexicon if isinstance(lex,li.Article)]
        obj_nouns = [lex for lex in self.lexicon if isinstance(lex,li.Noun) 
                                                 and not hasattr(lex,'lmk')]
        attr_adjs =[lex for lex in self.lexicon if isinstance(lex,li.Adjective)]
        simplenps = []
        adjnps = []
        for obj_noun in obj_nouns:
            for art in arts:
                simplenps.append(st.ReferringExpression([
                                    art,
                                    st.NounPhrase([obj_noun])
                                 ]))
                if obj_noun.regex == 'table':
                    the_table = simplenps[-1]
                for attr_adj in attr_adjs:
                    adjnps.append(st.ReferringExpression([
                                    art,
                                    st.AdjectiveNounPhrase([
                                        st.AdjectivePhrase([attr_adj]),
                                        obj_noun])
                                  ]))

        lmk_nouns = [lex for lex in self.lexicon if isinstance(lex,li.Noun) 
                                                 and hasattr(lex,'lmk')]
        dirs = [lex for lex in self.lexicon if isinstance(lex,li.Direction)]
        simplelmknps = []
        onedirection = []
        twodirection = []
        for lmk_noun in lmk_nouns:
            for art in arts:
                simplelmknps.append(
                    st.ExtrinsicReferringExpression([
                        art,
                        st.RelationNounPhrase([
                            st.NounPhrase([lmk_noun]),
                            st.RelationLandmarkPhrase([
                                st.PartOfRelation([li.of]),
                                the_table
                            ])
                        ])
                        
                    ]))
                for d in dirs:
                    onedirection.append(
                        st.ExtrinsicReferringExpression([
                            art,
                            st.RelationNounPhrase([
                                st.AdjectiveNounPhrase([
                                    st.AdjectivePhrase([
                                        st.OrientationAdjective([d])
                                    ]),
                                    lmk_noun
                                ]),
                                st.RelationLandmarkPhrase([
                                    st.PartOfRelation([li.of]),
                                    the_table
                                ])
                            ])
                            
                        ]))
                    if lmk_noun.regex == 'corner':
                        for d2 in dirs:
                            if d2 != d:
                                twodirection.append(
                                    st.ExtrinsicReferringExpression([
                                        art,
                                        st.RelationNounPhrase([
                                            st.AdjectiveNounPhrase([
                                                st.TwoAdjectivePhrase([
                                                    st.OrientationAdjective([d]),
                                                    st.OrientationAdjective([d2])
                                                ]),
                                                lmk_noun
                                            ]),
                                            st.RelationLandmarkPhrase([
                                                st.PartOfRelation([li.of]),
                                                the_table
                                            ])
                                        ])
                            
                        ]))

        # for snp in simplenps:
        #     print snp.prettyprint()
        # print
        # print
        # for anp in adjnps:
        #     print anp.prettyprint()

        # for slnp in simplelmknps:
        #     print slnp.prettyprint()
        # for od in onedirection:
        #     print od.prettyprint()
        # for td in twodirection:
        #     print td.prettyprint()

        self.landmark_parses = \
            simplenps+adjnps+simplelmknps+onedirection+twodirection


    def complete_The_object__parses(self):
        target = st.RelationNounPhrase
        pattern_start = [[st.NounPhrase([li.objct])]]
        current_depths = {st.ReferringExpression:1}
        rnpparses = apg.finish_parse(targetclass=target,
                                     pattern_start=pattern_start,
                                     lexicon=self.lexicon,
                                     structicon=self.structicon,
                                     current_depths=current_depths)
        parses = [st.ExtrinsicReferringExpression([li.the, parse]) 
                  for parse in rnpparses]
        self.The_object__parses = parses

    def weight_parses(self, referent, parses):
        potential_referents = self.context.get_all_potential_referents()
        scores = []
        for parse in parses:
            # utils.logger(parse.prettyprint())
            applicabilities = parse.sempole().ref_applicabilities(self.context, 
                                    potential_referents)
            # utils.logger(applicabilities)
            applicability = float(applicabilities[referent])
            # utils.logger(applicability)
            sumav = sum(applicabilities.values())
            if sumav == 0: old_score = 0
            else: old_score = (applicability**2) / sumav
            scores.append(old_score)
            # raw_input()

        result = sorted(zip(scores, parses), reverse=True)
        return result

    @staticmethod
    def choose_top_parse(sorted_parses):
        top_score = sorted_parses[0][0]
        # utils.logger(sorted_parses[:10])
        top_parses = []
        for score, parse in sorted_parses:
            if score == top_score:
                top_parses.append(parse)
            else:
                break
        # utils.logger(top_parses)
        return random.choice(top_parses)

    @staticmethod
    def sample_parse(sorted_parses):
        raise NotImplementedError

    def create_relation_memories(self, parse, true_referent):
        # Making all kinds of assumptions here
        assert(len(true_referent)==1)
        true_referent = true_referent[0]
        result = ''
        partials = parse.current[0].find_partials()
        utils.logger(partials[0])
        assert(len(partials) == 1)
        partial = partials[0]
        assert(partial.construction == st.RelationLandmarkPhrase)
        holes = partial.get_holes()
        assert(len(holes) == 1)
        hole = holes[0]
        relatum_constraints = partial.constituents[1].sempole()
        # result += str(relatum_constraints)
        result += hole.prettyprint()
        relata_apps = self.context.get_potential_referent_scores()
        relata_apps *= relatum_constraints.ref_applicabilities(self.context, 
                            relata_apps.keys())
        construction = 'Relation'
        leaves = []
        for seq in hole.unmatched_sequence:
            leaves.extend(seq.collect_leaves())
        string = ' '.join(leaves)

        threshold = 0.25
        for relatum, relatum_app in relata_apps.items():
            if relatum_app > threshold:
                assert(len(relatum)==1)
                relatum = relatum[0]

                # First good examples
                kwargs = {'construction':construction,
                          'string':string,
                          'exemplary':True,
                          'lmk_prob':relatum_app,
                          }
                for feature in g2f.feature_list:
                    kwargs[feature.domain.name] = \
                        feature.domain.datatype(
                            feature.observe(referent=true_referent,
                                            relatum=relatum,
                                            context=self.context))
                new_key = self.connection.execute(self.unknown_structs.insert(), 
                                                  scene_id=self.scene_key, 
                                                  **kwargs
                                                  ).inserted_primary_key[0]
                result += 'Inserted positive observation %i\n' % new_key

                # Now bad examples
                for false_referent in self.context.get_potential_referents():
                    assert(len(false_referent)==1)
                    false_referent = false_referent[0]
                    if false_referent != true_referent:
                        kwargs = {'construction':construction,
                                  'string':string,
                                  'exemplary':False,
                                  'lmk_prob':relatum_app,
                                  }
                        for feature in g2f.feature_list:
                            kwargs[feature.domain.name] = \
                                feature.domain.datatype(
                                    feature.observe(referent=false_referent,
                                                    relatum=relatum,
                                                    context=self.context))
                        new_key = self.connection.execute(
                                                self.unknown_structs.insert(), 
                                                scene_id=self.scene_key, 
                                                **kwargs
                                                ).inserted_primary_key[0]
                        result += 'Inserted negative observation %i\n' % new_key

        return result

    circ_prob_funcs = [pf.DecayEnvelope, pf.LogisticBell]

    def create_relation_construction(self, parse):
        result = ''
        partials = parse.current[0].find_partials()
        assert(len(partials) == 1)
        partial = partials[0]
        assert(partial.construction == st.RelationLandmarkPhrase)
        holes = partial.get_holes()
        assert(len(holes) == 1)
        hole = holes[0]
        # result += str(relatum_constraints)
        construction_class = 'Relation'
        leaves = []
        for seq in hole.unmatched_sequence:
            leaves.extend(seq.collect_leaves())
        string = ' '.join(leaves)
        query = alc.sql.select([self.unknown_structs]).where(
                self.unknown_structs.c.construction==construction_class).where(
                self.unknown_structs.c.string==string)
        results = list(self.connection.execute(query))
        if len(results) > 0:
            separated = zip(*results)
            classes = np.array(separated[4])
            probs = np.array(separated[5])
            feature_values = separated[6:]
            # result = ''
            # result += str(probs)+'\n'
            # result += str(features)+'\n'
            for feature, values in zip(g2f.feature_list, feature_values):
                values = [float('nan') if v is None else v for v in values]
                values = np.array(values)
                result += 'Feature: %s\nValues: %s\nClasses: %s\nProbs: %s\n' % \
                            (feature.domain.name, values, classes, probs)
                # utils.logger(result)
                if isinstance(feature.domain, dom.CircularDomain):
                    for pfunc in self.circ_prob_funcs:
                        result += 'Prob Func: %s\n' % pfunc
                        values[np.where(values == None)] = np.nan
                        mu, std = pfunc.estimate_parameters(feature.domain, 
                                                            values, 
                                                            classes)
                        result += '  Estimated mu: %f, std: %f\n' % (mu,std)
                        trial_pfunc = pfunc(mu, std, feature.domain)
                        class_probs = trial_pfunc(values)
                        class_probs[np.where(np.isnan(class_probs))] = 0
                        result += '  Estimated probs: %s\n' % class_probs
                        error = trial_pfunc.binomial_error(class_probs, classes)
                        result += '  Binomial error: %f\n' % error
                elif isinstance(feature.domain, dom.NumericalDomain):
                    mu, std = pf.LogisticSigmoid.estimate_parameters(feature.domain,
                                                                     values,
                                                                     classes)
                    # trial_constraint = 

                elif isinstance(feature.domain, dom.DiscreteDomain):
                    pass


        return result
