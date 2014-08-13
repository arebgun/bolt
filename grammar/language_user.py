import random
import numpy as np
import operator as op
import utils
import domain as dom
import common as cmn
import lexical_items as li
import constructions as st
import construction
import partial_parser as pp
from all_parse_generator import AllParseGenerator as apg

import os
import sqlalchemy as alc
import sqlalchemy.orm as orm
import gen2_features as g2f
import probability_function as pf
import domain as dom
import constraint as cnstrnt
import sempoles

import inspect
import IPython
import random as rand
import itertools as it
import scipy.stats as stats
import collections as coll
import kde
import weighted_kde as wkde

import pprint

class LanguageUser(object):
    db_suffix = '_memories.db'

    def __init__(self, name, lexicon, structicon, meta,
                       remember=True, reset=False):
        self.name = name
        self.db_name = 'student_memories'#self.name + self.db_suffix
        self.lexicon = lexicon
        self.structicon = structicon
        self.meta_grammar = meta
        self.remember = remember

        # db_name = self.name + self.db_suffix
        # if reset and os.path.isfile(db_name):
        #     os.remove(self.name + self.db_suffix)
        if reset:
            self.create_memorybase()
            # self.create_memorybase_intrinsic()

    def create_memorybase(self):
        # engine = alc.create_engine('sqlite:///'+self.db_name, echo=False)
        engine = alc.create_engine('postgresql://postgres:postgres@localhost/student_memories', echo=False)
        metadata = alc.MetaData(bind=engine, reflect=True)
        metadata.drop_all()
        s = alc.Table('scenes', metadata,
            alc.Column('id', alc.Integer, primary_key=True),
            extend_existing=True)

        feature_columns = []
        for feature in g2f.feature_list:
            type_map = {bool:alc.Boolean, int:alc.Integer, 
                        float:alc.Float, str:alc.String}
            feature_columns.append(alc.Column(feature.domain.name, 
                                            type_map[feature.domain.datatype]))

        o = alc.Table('observations', metadata,
            alc.Column('id', alc.Integer, primary_key=True),
            alc.Column('scene_id', None, alc.ForeignKey('scenes.id')),
            # alc.Column('utterance', alc.String),
            *feature_columns,
            extend_existing=True)

        u = alc.Table('utterances', metadata,
            alc.Column('id', alc.Integer, primary_key=True),
            alc.Column('observation_id', None, alc.ForeignKey('observations.id')),
            alc.Column('utterance', alc.String),
            alc.Column('exemplary', alc.Boolean),
            alc.Column('baseline_lmk_prob', alc.Float),
            extend_existing=True)

        un = alc.Table('unknown_structs', metadata,
            alc.Column('id', alc.Integer, primary_key=True),
            alc.Column('utterance_id', None, alc.ForeignKey('utterances.id')),
            # alc.Column('scene_id', None, alc.ForeignKey('scenes.id')),
            alc.Column('construction', alc.String),
            alc.Column('string', alc.String),
            alc.Column('lmk_prob', alc.Float),
            # *feature_columns)
            extend_existing=True)

        metadata.create_all(engine)
        conn = engine.connect()
        conn.execute(un.delete())
        conn.execute(u.delete())
        conn.execute(o.delete())
        conn.execute(s.delete())
        conn.close()
        # IPython.embed()
        self.engine = engine
        self.meta = metadata

    # def create_memorybase_intrinsic(self):
    #     # engine = alc.create_engine('sqlite:///'+self.db_name, echo=False)
    #     engine = alc.create_engine('postgresql://postgres:postgres@localhost/intrinsic_memories', echo=False)
    #     metadata = alc.MetaData(bind=engine, reflect=True)
    #     metadata.drop_all()
    #     s = alc.Table('scenes', metadata,
    #         alc.Column('id', alc.Integer, primary_key=True),
    #         extend_existing=True)

    #     feature_columns = []
    #     for feature in g2f.feature_list:
    #         type_map = {bool:alc.Boolean, int:alc.Integer, 
    #                     float:alc.Float, str:alc.String}
    #         feature_columns.append(alc.Column(feature.domain.name, 
    #                                         type_map[feature.domain.datatype]))

    #     o = alc.Table('observations', metadata,
    #         alc.Column('id', alc.Integer, primary_key=True),
    #         alc.Column('scene_id', None, alc.ForeignKey('scenes.id')),
    #         # alc.Column('utterance', alc.String),
    #         *feature_columns,
    #         extend_existing=True)

    #     u = alc.Table('utterances', metadata,
    #         alc.Column('id', alc.Integer, primary_key=True),
    #         alc.Column('observation_id', None, alc.ForeignKey('observations.id')),
    #         alc.Column('utterance', alc.String),
    #         alc.Column('exemplary', alc.Boolean),
    #         alc.Column('baseline_lmk_prob', alc.Float),
    #         extend_existing=True)

    #     un = alc.Table('unknown_structs', metadata,
    #         alc.Column('id', alc.Integer, primary_key=True),
    #         alc.Column('utterance_id', None, alc.ForeignKey('utterances.id')),
    #         # alc.Column('scene_id', None, alc.ForeignKey('scenes.id')),
    #         alc.Column('construction', alc.String),
    #         alc.Column('string', alc.String),
    #         alc.Column('lmk_prob', alc.Float),
    #         # *feature_columns)
    #         extend_existing=True)

    #     metadata.create_all(engine)
    #     conn = engine.connect()
    #     conn.execute(un.delete())
    #     conn.execute(u.delete())
    #     conn.execute(o.delete())
    #     conn.execute(s.delete())
    #     conn.close()
    #     # IPython.embed()
    #     self.engine = engine
    #     self.meta = metadata

    # def get_engine(self):
    #     engine = alc.create_engine('sqlite:///'+self.db_name, echo=False)
    #     return engine

    def connect_to_memories(self):#, engine, meta):
        # utils.logger('Connecting to memories')
        self.engine = alc.create_engine('postgresql://postgres:postgres@localhost/student_memories', echo=False)
        # self.engine = engine
        # self.meta = meta
        self.meta = alc.MetaData()
        self.meta.reflect(bind=self.engine)
        # IPython.embed()
        self.scenes = self.meta.tables['scenes']
        self.observations = self.meta.tables['observations']
        self.utterances = self.meta.tables['utterances']
        self.unknown_structs = self.meta.tables['unknown_structs']
        # self.connection = engine.connect()

    def connect_to_memories_intrinsic(self):#, engine, meta):
        # utils.logger('Connecting to memories')
        self.iengine = alc.create_engine('postgresql://postgres:postgres@localhost/intrinsic_memories', echo=False)
        # self.engine = engine
        # self.meta = meta
        self.imeta = alc.MetaData()
        self.imeta.reflect(bind=self.engine)
        # IPython.embed()
        self.iscenes = self.meta.tables['scenes']
        self.iobservations = self.meta.tables['observations']
        self.iutterances = self.meta.tables['utterances']
        self.iunknown_structs = self.meta.tables['unknown_structs']
        # self.connection = engine.connect()

    def disconnect(self):
        self.engine.dispose()

    def query(self, query, *args, **kwargs):
        conn = self.engine.connect()
        result = conn.execute(query, *args, **kwargs)
        conn.close()
        return result

    def iquery(self, query, *args, **kwargs):
        conn = self.iengine.connect()
        result = conn.execute(query, *args, **kwargs)
        conn.close()
        return result

    def copy(self):
        l = LanguageUser(self.name, self.lexicon, self.structicon, 
                        self.meta_grammar, remember=self.remember, reset=False)
        # l.engine = self.engine
        # l.meta = self.meta
        return l

    def set_context(self, context, generate=True):
        # utils.logger(self.name)
        if self.remember:
            self.scene_key = self.query(self.scenes.insert()
                                ).inserted_primary_key[0]
        self.context = context
        if generate:
            self.generate_landmark_parses()
            self.complete_The_object__parses()


    def create_observations(self):
        prefs = self.context.get_all_potential_referents()
        self.obsv_keys = {}
        for ref in prefs:
            for rel in [(None,)] + prefs:
                self.obsv_keys[(ref, rel)] = self.create_observation(ref, rel)

    def create_observation(self, referent, relatum):
        kwargs = {'scene_id':self.scene_key}
        for feature in g2f.feature_list:
            kwargs[feature.domain.name] = \
                feature.observe(referent=referent[0],
                                relatum=relatum[0],
                                context=self.context)
        new_key = self.query(self.observations.insert(),  
                    **kwargs).inserted_primary_key[0]
        return new_key

    def choose_referent(self, sorted_parses, return_all_tied=False):
        tree = sorted_parses[0].current[0]
        # utils.logger(tree.prettyprint())
        return self.choose_from_tree(tree, return_all_tied)[0]

    def choose_from_tree(self, tree_root, return_all_tied=False):
        result = ''
        potential_referents = self.context.get_all_potential_referents()
        if tree_root.sempole().relatum_constraints is not None:
            relata_apps = tree_root.sempole().relatum_constraints.ref_applicabilities(self.context, 
                            potential_referents)
            result += 'Relata apps\n'
            result += str(coll.Counter(relata_apps))+'\n'

        apps = tree_root.sempole().ref_applicabilities(self.context, 
                                                   potential_referents,
                                                   separate=True)
        result += 'Referent apps\n'
        result += pprint.pformat(apps)+'\n'
        # utils.logger(pprint.pformat(apps))
        app_items = sorted(apps.items(), key=lambda x: [np.product(x[1]),np.sum(x[1])]+list(x[1]), reverse=True)
        best_app = app_items[0][1]
        best_refs = [ref for ref, app in app_items if app == best_app]
        if return_all_tied:
            return best_refs, result
        else:
            return random.choice(best_refs), result

    def parse(self, utterance, max_holes=1):
        lexical_parses = pp.get_all_lexical_parses(sentence=utterance, 
                                                   lexicon=self.lexicon)
        if max_holes < 1:
            lexical_parses = [lp for lp in lexical_parses if not any([isinstance(m.construction,construction.Unknown) for m in lp])]
        parses=pp.get_all_construction_parses(lexical_parses=lexical_parses,
                                              structicon=self.structicon,
                                              max_holes=max_holes)
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


        self.landmark_parses = \
            simplenps+adjnps+simplelmknps+onedirection+twodirection


    def complete_The_object__parses(self):

        rel_parses = apg.generate_parses(targetclass=st.OrientationRelation,
                                         lexicon=self.lexicon,
                                         structicon=self.structicon)
        rel_parses+= apg.generate_parses(targetclass=st.DistanceRelation,
                                         lexicon=self.lexicon,
                                         structicon=self.structicon)
        # rel_parses+= [li.on]

        rnpparses = []
        for refex in self.landmark_parses:
            for rel in rel_parses:
                parse = st.ExtrinsicReferringExpression([
                            li.the,
                            st.RelationNounPhrase([
                                st.NounPhrase([li.objct]),
                                st.RelationLandmarkPhrase([
                                    rel,
                                    refex
                                ])
                            ])
                        ])
                rnpparses.append(parse)
        self.The_object__parses = rnpparses

    def weight_parses(self, referent, parses):
        potential_referents = self.context.get_all_potential_referents()
        scores = []
        # num_parses = len(parses)
        for i, parse in enumerate(parses):
            # print '{0}\r'.format(i),
            # utils.logger('')
            # utils.logger('')
            # utils.logger('Parse %s of %s' % (i,num_parses))
            # utils.logger(referent)
            # utils.logger(parse.print_sentence())
            # utils.logger(parse.sempole())
            applicabilities = parse.sempole().ref_applicabilities(self.context, 
                                    potential_referents)
            # utils.logger(applicabilities[referent])
            applicability = float(applicabilities[referent])
            # utils.logger(applicability)
            sumav = sum(applicabilities.values())
            # utils.logger(sumav)
            if sumav == 0: old_score = 0
            # else: old_score = (applicability**1.25) / sumav
            else: old_score = (applicability**2) / sumav
            # utils.logger(old_score)
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

    # def create_relation_memories(self, parse, true_referent):
    #     # Making all kinds of assumptions here
    #     assert(len(true_referent)==1)
    #     true_referent = true_referent[0]
    #     result = ''
    #     partials = parse.current[0].find_partials()
    #     utils.logger(partials[0])
    #     assert(len(partials) == 1)
    #     partial = partials[0]
    #     assert(partial.construction == st.RelationLandmarkPhrase)
    #     holes = partial.get_holes()
    #     assert(len(holes) == 1)
    #     hole = holes[0]
    #     relatum_constraints = partial.constituents[1].sempole()
    #     # result += str(relatum_constraints)
    #     result += hole.prettyprint()
    #     relata_apps = self.context.get_potential_referent_scores()
    #     relata_apps *= relatum_constraints.ref_applicabilities(self.context, 
    #                         relata_apps.keys())
    #     construction = 'Relation'
    #     leaves = []
    #     for seq in hole.unmatched_sequence:
    #         leaves.extend(seq.collect_leaves())
    #     string = ' '.join(leaves)

    #     threshold = 0.25
    #     for relatum, relatum_app in relata_apps.items():
    #         if relatum_app > threshold:
    #             assert(len(relatum)==1)
    #             relatum = relatum[0]

    #             # First good examples
    #             kwargs = {'construction':construction,
    #                       'string':string,
    #                       'exemplary':True,
    #                       'lmk_prob':relatum_app,
    #                       }
    #             for feature in g2f.feature_list:
    #                 kwargs[feature.domain.name] = \
    #                     feature.domain.datatype(
    #                         feature.observe(referent=true_referent,
    #                                         relatum=relatum,
    #                                         context=self.context))
    #             new_key = self.connection.execute(self.unknown_structs.insert(), 
    #                                               scene_id=self.scene_key, 
    #                                               **kwargs
    #                                               ).inserted_primary_key[0]
    #             result += 'Inserted positive observation %i\n' % new_key

    #             # Now bad examples
    #             for false_referent in self.context.get_potential_referents():
    #                 assert(len(false_referent)==1)
    #                 false_referent = false_referent[0]
    #                 if false_referent != true_referent:
    #                     kwargs = {'construction':construction,
    #                               'string':string,
    #                               'exemplary':False,
    #                               'lmk_prob':relatum_app,
    #                               }
    #                     for feature in g2f.feature_list:
    #                         kwargs[feature.domain.name] = \
    #                             feature.domain.datatype(
    #                                 feature.observe(referent=false_referent,
    #                                                 relatum=relatum,
    #                                                 context=self.context))
    #                     new_key = self.connection.execute(
    #                                             self.unknown_structs.insert(), 
    #                                             scene_id=self.scene_key, 
    #                                             **kwargs
    #                                             ).inserted_primary_key[0]
    #                     result += 'Inserted negative observation %i\n' % new_key

    #     return result

    circ_prob_funcs = [pf.DecayEnvelope, pf.LogisticBell]

    # def create_relation_construction(self, parse):
    #     result = ''
    #     partials = parse.current[0].find_partials()
    #     assert(len(partials) == 1)
    #     partial = partials[0]
    #     assert(partial.construction == st.RelationLandmarkPhrase)
    #     holes = partial.get_holes()
    #     assert(len(holes) == 1)
    #     hole = holes[0]
    #     # result += str(relatum_constraints)
    #     construction_class = 'Relation'
    #     leaves = []
    #     for seq in hole.unmatched_sequence:
    #         leaves.extend(seq.collect_leaves())
    #     string = ' '.join(leaves)
    #     query = alc.sql.select([self.unknown_structs]).where(
    #             self.unknown_structs.c.construction==construction_class).where(
    #             self.unknown_structs.c.string==string)
    #     conn = self.engine.connect()
    #     results = list(conn.execute(query))
    #     conn.close()
    #     if len(results) > 0:
    #         separated = zip(*results)
    #         IPython.embed()
    #         exit()
    #         classes = np.array(separated[4])
    #         probs = np.array(separated[5])
    #         feature_values = separated[6:]
    #         # result = ''
    #         # result += str(probs)+'\n'
    #         # result += str(features)+'\n'
    #         for feature, values in zip(g2f.feature_list, feature_values):
    #             values = [float('nan') if v is None else v for v in values]
    #             values = np.array(values)
    #             result += 'Feature: %s\nValues: %s\nClasses: %s\nProbs: %s\n' % \
    #                         (feature.domain.name, values, classes, probs)
    #             # utils.logger(result)
    #             if isinstance(feature.domain, dom.CircularDomain):
    #                 for pfunc in self.circ_prob_funcs:
    #                     result += 'Prob Func: %s\n' % pfunc
    #                     values[np.where(values == None)] = np.nan
    #                     mu, std = pfunc.estimate_parameters(feature.domain, 
    #                                                         values, 
    #                                                         classes)
    #                     result += '  Estimated mu: %f, std: %f\n' % (mu,std)
    #                     trial_pfunc = pfunc(mu, std, feature.domain)
    #                     class_probs = trial_pfunc(values)
    #                     class_probs[np.where(np.isnan(class_probs))] = 0
    #                     result += '  Estimated probs: %s\n' % class_probs
    #                     error = trial_pfunc.binomial_error(class_probs, classes)
    #                     result += '  Binomial error: %f\n' % error
    #             elif isinstance(feature.domain, dom.NumericalDomain):
    #                 mu, std = pf.LogisticSigmoid.estimate_parameters(feature.domain,
    #                                                                  values,
    #                                                                  classes)
    #                 # trial_constraint = 

    #             elif isinstance(feature.domain, dom.DiscreteDomain):
    #                 pass


    #     return result

    def create_utterance_memories(self, utterance, true_referent, lmk_apps=None, cheating=False):
        utt_keys = {}
        for (ref, rel), id_key in self.obsv_keys.items():
            exemplary = (ref == true_referent)
            if cheating and (lmk_apps is not None):
                exemplary = exemplary and bool(lmk_apps[rel] == 1.0)
            # if cheating or lmk_apps[rel] > 0.001:
            utt_keys[(ref, rel)] = self.create_utterance_entry(utterance,
                                                              exemplary,
                                                              id_key,
                                                              # 1.0)
                                                              1.0 if cheating else lmk_apps[rel])
        return utt_keys

    def create_utterance_entry(self, utterance, exemplary, obsv_key, lmk_prob):
        new_key = self.query(self.utterances.insert(),
                      utterance=utterance,
                      exemplary=exemplary,
                      observation_id=obsv_key,
                      baseline_lmk_prob=lmk_prob).inserted_primary_key[0]
        return new_key


    def create_new_construction_memories(self, utt_keys, utterance, parses, goal_type, true_referent):
        result = ''
        already_done = []

        for parse in parses:
            assert(len(parse.current)==1)
            root = parse.current[0]
            cls = root.construction if isinstance(root, cmn.Match) else root.__class__
            if issubclass(cls,goal_type):
                # result += root.prettyprint()
                # Currently can only do single partial with single Hole
                partials = root.find_partials()
                assert(len(partials)==1)
                partial = partials[0]
                holes = partial.get_holes()
                assert(len(holes)==1)
                hole = holes[0]
                leaves = []
                for item in hole.unmatched_sequence:
                    leaves.extend(item.collect_leaves())
                const_string = ''.join(leaves)
                const_type = hole.unmatched_pattern
                signature = (const_type.__name__, const_string)
                if not signature in already_done:
                    already_done.append(signature)

                    refexes = []
                    for constituent in partial.constituents:
                        if isinstance(constituent,st.ReferringExpression):
                            refexes.append(constituent)
                    assert(len(refexes)<=1)

                    if len(refexes)==1:
                        relatum_constraints = refexes[0].sempole()

                        relata_apps = self.context.get_all_potential_referent_scores()
                        relata_apps *= relatum_constraints.ref_applicabilities(self.context, 
                                    relata_apps.keys())

                        result += str(signature)+'\n'
                        result += 'Landmark expression\n'
                        result += refexes[0].prettyprint()
                        result += '%s\n' % relatum_constraints
                        result += '%s\n' % relata_apps
                    else:
                        relata_apps = {(None,):1}

                    # if const_type in self.meta_grammar:
                        # hole._sempole = self.meta_grammar[const_type]()
                    hole._sempole = cnstrnt.ConstraintSet()
                    referent_apps = self.context.get_all_potential_referent_scores()
                    referent_apps *= root.sempole().ref_applicabilities(self.context,
                                    referent_apps.keys())
                    keys = []

                    # max_weight = 0
                    # for referent, ref_app in referent_apps.items():
                    #     for relatum, relatum_app in relata_apps.items():
                    #         weight = ref_app*relatum_app
                    #         if weight > max_weight:
                    #             max_weight = weight

                    # inverse_max_weight = 1-max_weight

                    for referent, ref_app in referent_apps.items():
                        if ref_app > 0:
                            assert(len(referent)==1)
                            for relatum, relatum_app in relata_apps.items():
                                if (referent,relatum) not in utt_keys:
                                    continue
                                assert(len(relatum)==1)
                                relatum = relatum
                                weight = ref_app*relatum_app# + inverse_max_weight
                                # result+= '%s==%s: %s\n' % (referent, true_referent, referent == true_referent)
                                if weight > 0:
                                    kwargs = {'utterance_id':utt_keys[(referent,relatum)],
                                              'construction':const_type.__name__,
                                              'string':const_string,
                                              'lmk_prob':weight,
                                              }
                                    new_key = self.query(self.unknown_structs.insert(), 
                                                                      **kwargs
                                                                      ).inserted_primary_key[0]
                                    # new_key=self.create_datapoint(referent == true_referent, 
                                    #                               utterance,
                                    #                               const_type.__name__, 
                                    #                               const_string, 
                                    #                               referent[0], relatum, 
                                    #                               weight)
                                    keys.append(new_key)
                    result += 'Inserted %s keys\n' % len(keys)

        return result

    # def create_datapoint(self, exemplary, utterance,construction, string, 
    #                      referent, relatum, relatum_app):
    #     kwargs = {'utterance':utterance,
    #               'construction':construction,
    #               'string':string,
    #               'exemplary':exemplary,
    #               'lmk_prob':relatum_app,
    #               }
    #     for feature in g2f.feature_list:
    #         kwargs[feature.domain.name] = \
    #             feature.observe(referent=referent,
    #                             relatum=relatum,
    #                             context=self.context)
    #             # feature.domain.datatype(
    #             #     feature.observe(referent=referent,
    #             #                     relatum=relatum,
    #             #                     context=self.context))
    #     new_key = self.connection.execute(self.unknown_structs.insert(), 
    #                                       scene_id=self.scene_key, 
    #                                       **kwargs
    #                                       ).inserted_primary_key[0]
    #     # result += 'Inserted positive observation %i\n' % new_key
    #     return new_key


    def construct_from_parses(self, parses, goal_type):#, goal_sem):
        result = ''
        completed = []
        for parse in parses[:1]:
            assert(len(parse.current)==1)
            root = parse.current[0]
            cls = root.construction if isinstance(root, cmn.Match) else root.__class__
            if issubclass(cls,goal_type):
                # result += root.prettyprint()
                partials = root.find_partials()
                assert(len(partials)==1)
                partial = partials[0]
                holes = partial.get_holes()
                assert(len(holes)==1)
                hole = holes[0]
                leaves = []
                for item in hole.unmatched_sequence:
                    leaves.extend(item.collect_leaves())
                const_string = ''.join(leaves)
                const_type = hole.unmatched_pattern
                # result += 'In the metagrammar\n' if const_type in self.meta_grammar else 'Not in meta\n'
                # if const_type in self.meta_grammar:
                result += '%s %s\n' % (const_type, const_string)
                # new_sempole = self.meta_grammar[const_type]()

                # IPython.embed()
                # constraint_set, result1 = self.old_construct(
                #                             const_type.__name__,
                #                             const_string)
                constraint_set, result1 = self.cv_build(
                                            const_type.__name__,
                                            const_string)
                result += result1
                if constraint_set:
                    hole._sempole = constraint_set
                    completed.append(root)
                    # utils.logger(root.prettyprint())
                    result += str(root.sempole())
                result += '\n'


        return completed, result

    @staticmethod
    def binomial_cost(probs, labels, weights):
        notlabels = np.logical_not(labels)
        w = np.where(np.logical_not(np.isnan(probs)))
        return 1-np.dot(labels[w]*probs[w]+notlabels[w]*(1-probs[w]), 
                        weights[w])/weights[w].sum()

    def cv_build(self, const_type, const_string, num_functions=3, num_folds=10):
        query = alc.sql.select(
                [self.utterances, self.unknown_structs, self.observations],
                use_labels=True
            ).where(self.observations.c.id==self.utterances.c.observation_id
            ).where(self.utterances.c.id==self.unknown_structs.c.utterance_id
            ).where(
                self.unknown_structs.c.construction==const_type).where(
                self.unknown_structs.c.string==const_string
            )

        results = list(self.query(query))
        if len(results) == 0:
            return None, 'First sighting of "%s --> %s"\n' % (const_type,const_string)
        #else:

        result = ''
        separated = zip(*results)
        classes = np.array(separated[3])
        weights = np.array(separated[9])
        feature_values = np.array(separated[12:]).T

        numpos = sum(classes)
        if numpos < 2:
            fold_csets,_ = self.construct(g2f.feature_list,
                                        feature_values,
                                        classes,
                                        weights,
                                        how_many = 1)
            if len(fold_csets) == 1:
                return fold_csets[0], ''
            else:
                return None, ''
        else:
            num_folds = min(numpos,num_folds)
        # utils.logger('num_folds %s' % num_folds)
        # utils.logger('len weights %s' % len(weights))

        result += 'Classes: %s\n' % classes
        result += 'Weights: %s\n' % weights

        num_funcs = range(1,num_functions+1)
        folds_indices = self.select_folds(classes, num_folds)
        if folds_indices is None:
            return None, ''
        # utils.logger(folds_indices)

        fold_scores = np.zeros((num_folds, len(num_funcs)))
        for i in range(num_folds):
            # utils.logger('fold %s' % i)
            test_ind = folds_indices[i]
            train_ind = list(it.chain(*(folds_indices[:i]+folds_indices[i+1:])))
            # utils.logger('fold train size %s' % len(train_ind))

            fold_csets,_ = self.construct(g2f.feature_list,
                                        feature_values[train_ind],
                                        classes[train_ind],
                                        weights[train_ind],
                                        how_many = num_functions)

            if fold_csets == []:
                continue
            # utils.logger(fold_csets)
            # shortcut, the last cset has all the functions, so have it 
            # return probs from 1st, 1st and 2nd, etc
            temp = fold_csets[-1]
            test_features = feature_values[test_ind]
            fold_probs = temp.judge_array(g2f.feature_list, test_features)
            for j, probs in enumerate(fold_probs):
                fold_scores[i,j] = self.binomial_cost(probs, 
                                                      classes[test_ind], 
                                                      weights[test_ind])

        utils.logger("\n%s" % fold_scores)
        mean_scores = stats.nanmean(fold_scores,axis=0)
        # utils.logger(mean_scores)
        utils.logger('Mean scores: %s' % mean_scores)
        min_ind = mean_scores.argmin()
        utils.logger('Best score: %s %s' % (min_ind, mean_scores[min_ind]))

        result,_ = self.construct(g2f.feature_list,
                              feature_values,
                              classes,
                              weights,
                              how_many = min_ind+1)
        # utils.logger(g2f.feature_list)
        return result[-1], ''

    @staticmethod
    def select_folds(labels, num_folds):
        w = list(np.where(labels)[0])
        nw = list(np.where(1-labels)[0])
        wper_fold = int(len(w)/num_folds)
        nwper_fold = int(len(nw)/num_folds)
        # utils.logger(w)
        # utils.logger(nw)
        # utils.logger(wper_fold)
        # utils.logger(nwper_fold)
        utils.logger('%s %s %s %s' % (len(labels), 
                                      wper_fold*num_folds, 
                                      nwper_fold*num_folds,
                                      wper_fold*num_folds+nwper_fold*num_folds))
        # indices = np.arange(num_instances)
        rand.shuffle(w)
        rand.shuffle(nw)
        if wper_fold == 0 or nwper_fold == 0:
            return None
        wchunks = chunks(w, wper_fold)
        nwchunks = chunks(nw, nwper_fold)
        result = [w+nw for w,nw in zip(wchunks,nwchunks)]

        return result
    def old_construct(self, const_type, const_string):
        query = alc.sql.select(
                [self.utterances, self.unknown_structs, self.observations],
                use_labels=True
            ).where(self.observations.c.id==self.utterances.c.observation_id
            ).where(self.utterances.c.id==self.unknown_structs.c.utterance_id
            ).where(
                self.unknown_structs.c.construction==const_type).where(
                self.unknown_structs.c.string==const_string
            )

        results = list(self.query(query))
        if len(results) == 0:
            return None, 'First sighting of "%s --> %s"\n' % (const_type,const_string)
        else:
            result = '%s memories for "%s --> %s"\n' % (len(results),const_type,const_string)

        separated = zip(*results)
        classes = np.array(separated[3])
        weights = np.array(separated[9])
        feature_values = np.array(separated[12:]).T
        constraint, fi, result1 = self.build_constraint(g2f.feature_list,
                                                            feature_values,
                                                            classes,
                                                            weights)
        # result += result1
        if constraint:
            return cnstrnt.ConstraintSet([constraint]), result
        else:
            return None, result

    def construct(self,feature_list,feature_values,classes,weights,how_many=1):
        feature_list = list(feature_list)
        result = ''

        # w = np.where(1-classes)
        constraints = []
        weights = np.array(weights)
        # a = zip(feature_list,zip(*feature_values))
        for _ in range(min(how_many,len(feature_list))):
            c, fi, res = self.build_constraint(feature_list, feature_values,
                                               classes, weights)
            # x =sorted(zip(classes, weights, a[7][1], a[8][1]))
            # utils.logger(_)
            # utils.logger(c)
            # IPython.embed()
            if c is None:
                break
            constraints.append(c)
            result += res
            feature_list[fi] = None
            weights *= cnstrnt.ConstraintSet([c]).judge_array(g2f.feature_list,feature_values)[-1]



        l = len(constraints)
        return [cnstrnt.ConstraintSet(constraints[:i]) for i in range(1,l+1)], \
               result

    def build_constraint(self,feature_list,feature_values,classes,weights):
        result = ''
        possible = []
        for fi, (feature, values) in enumerate(zip(feature_list, feature_values.T)):
            if feature is None:
                continue
            # utils.logger(feature)
            # utils.logger(values[:10])
            # raw_input()
            # values = np.array(values)
            result += '%s positive Values: %s\n' %(feature,values[np.where(classes==True)])
            result += '%s negative Values: %s\n' %(feature,values[np.where(classes==False)])


            if isinstance(feature.domain,dom.DiscreteDomain):
                # utils.logger('  discrete')
                # utils.logger(values)
                values = np.array(['None' if v is None else v for v in values])
                # pfunc, result1=pf.DiscreteProbFunc.build_binary(values[wnc],
                #                                        classes[wnc],weights[wnc],
                #                                        self.binomial_cost)
                # utils.logger(values)
                pfunc, result1=pf.DiscreteProbFunc.build_binary(values,
                                   classes,weights,
                                   self.binomial_cost)
                # utils.logger(pfunc)
                # utils.logger(result1)
                result += result1
                if pfunc:
                    cost = self.binomial_cost(pfunc(values),
                                              classes,
                                              weights)
                    result += '%s: %s, %s\n' % (feature,
                                                cost,
                                                pfunc)
                    # utils.logger('%s: %s, %s\n' % (feature,cost,pfunc))
                    possible.append((cost,pfunc,feature,fi))
            else:
                # utils.logger('  continuous')
                # if len(filter(None,values)) == 0:
                #     continue
                values = values.astype(float)
                # values = np.array([float('nan') if v is None else v for v in values])
                # w = np.where(np.logical_and(np.logical_not(np.isnan(values)),nc))
                # w = np.where(np.logical_not(np.isnan(values)))
                w = np.where(np.isfinite(values))
                classes_ = classes[w]
                values_ = values[w]
                weights_ = weights[w]
                pfunc=pf.LogisticBell.build(feature.domain,
                                            values_,
                                            classes_,
                                            weights_)
                if pfunc:
                    cost = self.binomial_cost(pfunc(values),
                                              classes,
                                              weights)
                    result += '%s: %s, %s\n' % (feature,
                                                cost,
                                                pfunc)
                    # utils.logger('%s: %s, %s\n' % (feature,cost,pfunc))
                    possible.append((cost,pfunc,feature,fi))

                pfunc=pf.LogisticSigmoid.build(feature.domain,
                                               values_,
                                               classes_,
                                               weights_)
                if pfunc:
                    cost = self.binomial_cost(pfunc(values),
                                              classes,
                                              weights)
                    result += '%s: %s, %s\n' % (feature,
                                                cost,
                                                pfunc)
                    # utils.logger('%s: %s, %s\n' % (feature,cost,pfunc))
                    possible.append((cost,pfunc,feature,fi))

        if len(possible) < 1:
            return None, 0, 'No constraints found'
        else:
            possible.sort()
            _, pfunc, feature, fi = possible[0]
            if 'relatum' in inspect.getargspec(feature.observe).args:
                constraint = cnstrnt.RelationConstraint(feature,pfunc)
            else:
                constraint = cnstrnt.PropertyConstraint(feature,pfunc)
            return constraint, fi, result

    def baseline1(self, utterance, lmk_apps=coll.Counter()):
        prefs = self.context.get_all_potential_referents()
        words = utterance.split()
        lmks = [(None,)] + prefs
        # pref_scores = dict([((ref, lmk), 1.0) for ref in prefs for lmk in lmks])
        reflmks = [(ref, lmk) for ref in prefs for lmk in lmks]
        pfeatwords = dict([((ref, lmk), np.log(lmk_apps[lmk])) for (ref, lmk) in reflmks])
        # pmis = dict([((ref, lmk), np.log(lmk_apps[lmk])) for (ref, lmk) in reflmks])
        # bayess = dict([((ref, lmk), np.log(lmk_apps[lmk])) for (ref, lmk) in reflmks])

        # query = alc.sql.select(
        #     [self.observations],
        #     use_labels=True
        # )
        # results = list(self.query(query))
        # separated = zip(*results)
        # all_feature_values = np.array(separated[2:])

        # uncon_dists = []
        # for feat, all_values in zip(g2f.feature_list, all_feature_values):
        #     if isinstance(feat.domain,dom.DiscreteDomain):
        #         uncondist = self.get_normed_hist(all_values)
        #     else: 
        #         uncondist = self.get_normed_pdf(all_values)
        #     # uncon_dists.append(uncondist)
        #     fvalue = feat.observe(referent=ref[0], relatum=lmk[0], context=self.context)
        #     px = uncondist(fvalue)
        #     pmis[(ref,lmk)] -= np.log(px)
        #     bayess[(ref,lmk)] -= np.log(px)

        # query = alc.sql.select([self.utterances])
        # query = alc.sql.select([alc.sql.func.count()], 
        #     from_obj=[query.alias('subquery')])

        # total_utterances = float(self.query(query).scalar())
        # if total_utterances == 0: total_utterances = 1

        for word in words:
            # query = alc.sql.select([self.utterances]).where(
            #     self.utterances.c.utterance.like('%%%s%%' % word)
            # )
            # query = alc.sql.select([alc.sql.func.count()], 
            #     from_obj=[query.alias('subquery')])

            # with_this_word = self.query(query).scalar()

            query = alc.sql.select(
                [self.utterances, self.observations],
                use_labels=True
            ).where(self.observations.c.id==self.utterances.c.observation_id
            ).where(
                self.utterances.c.utterance.like('%%%s%%' % word)
            )

            results = list(self.query(query))
            if len(results) == 0:
                continue

            separated = zip(*results)
            classes = np.array(separated[3])
            weights = np.array(separated[4])
            w = np.where(np.logical_and(classes, weights>0.001))
            # w = np.where(classes)
            # probs = np.array(separated[6])
            feature_values = np.array(separated[7:])
        
            # with_this_word = sum(classes)

            # bayess[(ref,lmk)] += np.log(with_this_word/total_utterances)
            # utils.logger('===== %s =====' % word)

            # if with_this_word == 0: with_this_word = 0.00000000001
            for feat, values in zip(g2f.feature_list, feature_values):
                if isinstance(feat.domain,dom.DiscreteDomain):
                    condist = self.get_normed_hist(values[w], weights=weights[w])
                else: 
                    condist = self.get_normed_pdf(values[w], weights=weights[w])

                for ref, lmk in reflmks:
                    fvalue = feat.observe(referent=ref[0], relatum=lmk[0], context=self.context)
                    # px = uncondist(fvalue)
                    pxy = condist(fvalue)
                    # print feat, ref, lmk, fvalue, d
                    # utils.logger((feat,ref,lmk,fvalue,pxy))
                    pfeatword = np.log(pxy)
                    # utils.logger(pfeatword)
                    # pmi = pfeatword-np.log(px)
                    # bayes = pmi+np.log(with_this_word/total_utterances)
                    pfeatwords[(ref,lmk)] += pfeatword
                    # pmis[(ref,lmk)] += pfeatword
                    # bayess[(ref,lmk)] += pfeatword
                    # pref_scores[(ref,lmk)] += (np.log(pxy)
                    #                            +np.log(with_this_word/total_utterances)
                    #                            -np.log(px))
            # utils.logger('')
        # IPython.embed()
        result = ''
        items = [(p, reflmk) for (reflmk, p) in pfeatwords.items() if np.isfinite(p)]
        if len(items) < 1:
            return None, result
        items = sorted(items, reverse=True)
        pfeatword_choice = items[0][1][0]
        for item in items[:10]:
            result += str(item)+'\n'
        # pmi_choice = sorted(pmis.items(), key=op.itemgetter(1), reverse=True)[0][0][0]
        # bayes_choice = sorted(bayess.items(), key=op.itemgetter(1), reverse=True)[0][0][0]

        # exit()

        return pfeatword_choice, result #, pmi_choice, bayes_choice, ''

    def baseline_prep(self, utterance):
        # prefs = self.context.get_all_potential_referents()
        # words = utterance.split()
        # lmks = prefs
        # pfeatwords = dict([((ref,lmk), 0) for lmk in lmks])
        for ref in self.context.get_all_potential_referents():
            if ref[0].name == 'table':
                table = ref

        prefs = self.context.get_all_potential_referents()
        words = utterance.split()
        lmks = [(None,),table]
        # pref_scores = dict([((ref, lmk), 1.0) for ref in prefs for lmk in lmks])
        reflmks = [(ref, lmk) for ref in prefs for lmk in lmks]
        pfeatwords = dict([((ref, lmk), 0) for (ref, lmk) in reflmks])

        for word in words:
            # query = alc.sql.select([self.utterances]).where(
            #     self.utterances.c.utterance.like('%%%s%%' % word)
            # )
            # query = alc.sql.select([alc.sql.func.count()], 
            #     from_obj=[query.alias('subquery')])

            # with_this_word = self.query(query).scalar()

            string = '%%\y%s\y%%' % word
            # utils.logger(string)
            query = alc.sql.select(
                [self.iutterances, self.iobservations],
                use_labels=True
            ).where(self.iobservations.c.id==self.iutterances.c.observation_id
            ).where(
                self.iutterances.c.utterance.op("SIMILAR TO")(string)
            )

            results = list(self.iquery(query))
            if len(results) == 0:
                # utils.logger('CONTINUING: %s' % word)
                continue

            separated = zip(*results)
            classes = np.array(separated[3])
            # weights = np.array(separated[4])
            # w = np.where(np.logical_and(classes, weights>0))
            w = np.where(classes)
            # probs = np.array(separated[6])
            feature_values = np.array(separated[7:])
        
            with_this_word = sum(classes)
            # utils.logger('===== %s =====' % word)
            if with_this_word == 0: with_this_word = 0.00000000001
            for feat, values in zip(g2f.feature_list, feature_values):
                if isinstance(feat.domain,dom.DiscreteDomain):
                    condist = self.get_normed_hist(values[w])
                else: 
                    condist = self.get_normed_pdf(values[w])

                # for lmk in lmks:
                for ref, lmk in reflmks:
                    fvalue = feat.observe(referent=ref[0], relatum=lmk[0], context=self.context)
                    pxy = condist(fvalue)
                    if pxy == 0: pxy = 0.00000000001
                    # utils.logger((feat,ref,lmk,fvalue,pxy))
                    pfeatword = np.log(pxy)
                    # utils.logger(pfeatword)
                    pfeatwords[(ref,lmk)] += pfeatword
            # utils.logger('')

        result = ''
        items = [[p, reflmk] for (reflmk, p) in pfeatwords.items() if np.isfinite(p)]
        if len(items) < 1:
            return None, result
        items = sorted(items, reverse=True)
        for item in items[:10]:
            result += str(item)+'\n'

        for item in items:
            item[0] = np.exp(item[0])
        total = sum(zip(*items)[0])
        for item in items:
            item[0] /= total
        # for key, value in pfeatwords.items():
        #     pfeatwords[key] /= total
        #     # if pfeatwords[key] < 0.001:
        #     #     del pfeatwords[key]

        to_return = coll.Counter(dict([(ref, 0) for ref in prefs]))
        for value, (key,_) in items:
            if value > to_return[key]:
                to_return[key] = value

        return to_return, result

    def get_normed_hist(self, values, weights=None):
        if weights is not None:
            countxy = coll.defaultdict(float)
            weights /= float(sum(weights))
            for value, weight in zip(values, weights):
                countxy[value] += weight
        else:
            countxy = coll.Counter(values)
            totalxy = float(len(values))#float(sum(countxy.values()))
            for key in countxy:
                countxy[key] /= totalxy
        # epsilon = 0#.00000000001
        def hist(value):
            # if countx[value] == 0:
            #     return 0.00000000001, 0.00000000001
            return countxy[value]#+epsilon#, countx[value]
        return hist

    def get_normed_pdf(self, values, weights=None):
        if weights is None:
            weights = np.array([1.0]*len(values))
        valuesxy = np.array(values, dtype=float)
        # valuesxy = valuesxy[np.where(classes)]
        w = np.where(np.isfinite(valuesxy))
        not_nonexy = valuesxy[w]
        not_noneweights = weights[w]
        totalxy = float(sum(weights))
        # totalxy = float(len(valuesxy))


        if len(not_nonexy) < 2 or sum(not_noneweights) < 0.001:
            def pdf(value):
                # utils.logger('Fake pdf')
                if value is None or not np.isfinite(value):
                    return 1
                else:
                    return 0
            return pdf

        # if all(not_nonexy == not_nonexy[0]):
        #     return self.get_normed_hist(values, weights)

        frac_nonexy = (totalxy-sum(not_noneweights))/totalxy
        # frac_nonexy = (totalxy-len(not_nonexy))/totalxy
        # if frac_nonexy == 0: frac_nonexy = 0.00000000001
        kernelxy = kde.kde(not_nonexy, weights=not_noneweights)
        if kernelxy is None:
            # if weights is None:
            # kernelxy = stats.gaussian_kde(not_nonexy)
            # else:
            # utils.logger(not_nonexy)
            # utils.logger(not_noneweights)
            kernelxy = wkde.gaussian_kde(not_nonexy,weights=not_noneweights)

        # if any(np.isnan(kernelxy(not_nonexy))):
        #     utils.logger('Returning hist')
        #     return self.get_normed_hist(values, weights)

        def pdf(value):
            if value is None or np.isnan(value):
                return frac_nonexy#, frac_nonex
            # return kernelxy(value)[0]*(1-frac_nonexy), (kernelx(value)[0]*(1-frac_nonex))
            k = kernelxy(value)[0]
            if np.isnan(k):
                return (1-frac_nonexy)
            return k*(1-frac_nonexy)#, kernelx(value)[0]*(1-frac_nonex)
        return pdf

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
