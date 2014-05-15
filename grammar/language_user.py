import random
import numpy as np
import operator as op
import utils
import domain as dom
import common as cmn
import lexical_items as li
import constructions as st
import partial_parser as pp
from all_parse_generator import AllParseGenerator as apg

import os
import sqlalchemy as alc
import gen2_features as g2f
import probability_function as pf
import domain as dom
import constraint as cnstrnt
import sempoles

import inspect
import IPython
import random as rand
import itertools as it
from scipy.stats.stats import nanmean

class LanguageUser(object):
    db_suffix = '_memories.db'

    def __init__(self, name, lexicon, structicon, meta,
                       remember=True, reset=False):
        self.name = name
        self.db_name = self.name + self.db_suffix
        self.lexicon = lexicon
        self.structicon = structicon
        self.meta_grammar = meta
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
            alc.Column('utterance', alc.String),
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
                        self.meta_grammar, remember=self.remember, reset=False)

    def set_context(self, context):
        utils.logger(self.name)
        if self.remember:
            self.scene_key = self.connection.execute(self.scenes.insert()
                                ).inserted_primary_key[0]
        self.context = context
        self.generate_landmark_parses()
        self.complete_The_object__parses()

    def choose_referent(self, sorted_parses, return_all_tied=False):
        tree = sorted_parses[0].current[0]
        return self.choose_from_tree(tree, return_all_tied)

    def choose_from_tree(self, tree_root, return_all_tied=False):
        potential_referents = self.context.get_all_potential_referents()
        apps = tree_root.sempole().ref_applicabilities(self.context, 
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
            # utils.logger('Parse %s of %s' % (i,num_parses))
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
            IPython.embed()
            exit()
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

    def create_new_construction_memories(self, utterance, parses, goal_type, true_referent):
        result = ''
        already_done = []
        for parse in parses:
            assert(len(parse.current)==1)
            root = parse.current[0]
            cls = root.construction if isinstance(root, cmn.Match) else root.__class__
            if issubclass(cls,goal_type):
                result += root.prettyprint()
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
                const_string = ' '.join(leaves)
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
                    else:
                        relata_apps = {(None,):1}

                    if const_type in self.meta_grammar:
                        hole._sempole = self.meta_grammar[const_type]()
                        referent_apps = self.context.get_all_potential_referent_scores()
                        referent_apps *= root.sempole().ref_applicabilities(self.context,
                                        referent_apps.keys())
                        keys = []
                        for referent, ref_app in referent_apps.items():
                            if ref_app > 0:
                                assert(len(referent)==1)
                                for relatum, relatum_app in relata_apps.items():
                                    assert(len(relatum)==1)
                                    relatum = relatum[0]
                                    weight = ref_app*relatum_app
                                    # result+= '%s==%s: %s\n' % (referent, true_referent, referent == true_referent)
                                    if weight > 0:
                                        new_key=self.create_datapoint(referent == true_referent, 
                                                                      utterance,
                                                                      const_type.__name__, 
                                                                      const_string, 
                                                                      referent[0], relatum, 
                                                                      weight)
                                        keys.append(new_key)
                        result += 'Inserted %s keys\n' % len(keys)

        return result



    def create_datapoint(self, exemplary, utterance,construction, string, 
                         referent, relatum, relatum_app):
        kwargs = {'utterance':utterance,
                  'construction':construction,
                  'string':string,
                  'exemplary':exemplary,
                  'lmk_prob':relatum_app,
                  }
        for feature in g2f.feature_list:
            kwargs[feature.domain.name] = \
                feature.observe(referent=referent,
                                relatum=relatum,
                                context=self.context)
                # feature.domain.datatype(
                #     feature.observe(referent=referent,
                #                     relatum=relatum,
                #                     context=self.context))
        new_key = self.connection.execute(self.unknown_structs.insert(), 
                                          scene_id=self.scene_key, 
                                          **kwargs
                                          ).inserted_primary_key[0]
        # result += 'Inserted positive observation %i\n' % new_key
        return new_key


    def construct_from_parses(self, parses, goal_type):#, goal_sem):
        result = ''
        completed = []
        for parse in parses:
            assert(len(parse.current)==1)
            root = parse.current[0]
            cls = root.construction if isinstance(root, cmn.Match) else root.__class__
            if issubclass(cls,goal_type):
                result += root.prettyprint()
                partials = root.find_partials()
                assert(len(partials)==1)
                partial = partials[0]
                holes = partial.get_holes()
                assert(len(holes)==1)
                hole = holes[0]
                leaves = []
                for item in hole.unmatched_sequence:
                    leaves.extend(item.collect_leaves())
                const_string = ' '.join(leaves)
                const_type = hole.unmatched_pattern
                result += '%s %s\n' % (const_type, const_string)
                if const_type in self.meta_grammar:
                    # new_sempole = self.meta_grammar[const_type]()

                    # IPython.embed()
                    constraint_set, result1 = self.old_construct(
                                                const_type.__name__,
                                                const_string)
                    # constraint_set, result1 = self.cv_build(
                                                # const_type.__name__,
                                                # const_string)
                    result += result1
                    if constraint_set:
                        hole._sempole = constraint_set
                        completed.append(root)
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
        query = alc.sql.select([self.unknown_structs]).where(
            self.unknown_structs.c.construction==const_type).where(
            self.unknown_structs.c.string==const_string)
        results = list(self.connection.execute(query))
        if len(results) == 0:
            return None, 'First sighting of "%s --> %s"\n' % (const_type,const_string)
        #else:

        result = ''
        separated = zip(*results)
        classes = np.array(separated[5])
        weights = np.array(separated[6])
        feature_values = np.array(separated[7:]).T

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
        mean_scores = nanmean(fold_scores,axis=0)
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
        wchunks = chunks(w, wper_fold)
        nwchunks = chunks(nw, nwper_fold)
        result = [w+nw for w,nw in zip(wchunks,nwchunks)]

        return result
    def old_construct(self, const_type, const_string):
        query = alc.sql.select([self.unknown_structs]).where(
            self.unknown_structs.c.construction==const_type).where(
            self.unknown_structs.c.string==const_string)
        results = list(self.connection.execute(query))
        if len(results) == 0:
            return None, 'First sighting of "%s --> %s"\n' % (const_type,const_string)
        #else:

        separated = zip(*results)
        classes = np.array(separated[5])
        weights = np.array(separated[6])
        feature_values = np.array(separated[7:]).T
        constraint, fi, result1 = self.build_constraint(g2f.feature_list,
                                                            feature_values,
                                                            classes,
                                                            weights)
        if constraint:
            return cnstrnt.ConstraintSet([constraint]), result1
        else:
            return None, result1

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
                values = np.array(['None' if v is None else v for v in values])
                # pfunc, result1=pf.DiscreteProbFunc.build_binary(values[wnc],
                #                                        classes[wnc],weights[wnc],
                #                                        self.binomial_cost)
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
                if len(filter(None,values)) == 0:
                    continue
                values = values.astype(float)
                # values = np.array([float('nan') if v is None else v for v in values])
                # w = np.where(np.logical_and(np.logical_not(np.isnan(values)),nc))
                w = np.where(np.logical_not(np.isnan(values)))
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

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]