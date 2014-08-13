#!/usr/bin/python
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

import operator as op
# import matplotlib as mpl
import multiprocessing as mp

import sqlalchemy as alc
import sqlalchemy.orm as orm
import gen2_features as g2f

import os

import matplotlib.pyplot as plt

import IPython
import traceback
import shelve
import time
import numpy as np
import argparse
import random
import collections as coll

global training_data
# global answers
# answers = []


def one_scene(args):
    # global answers
    try:
        student, goal_type, cheating, learning, extrinsic, turk, i, test_i = args

        if i is not None:
            global training_data
            scene, speaker, refs_parses = training_data[i]
        else:
            global test_data
            scene, speaker, refs_parses = test_data[test_i]
        # utils.logger(scene)
        # utils.logger('Starting %s, %s' % (refs_parses[0][0],refs_parses[0][1][0].print_sentence()))
        utils.logger('Starting %s, %s' % (refs_parses[0][0],refs_parses[0][1].print_sentence()))
        context = cmn.Context(scene, speaker)
        # global engine
        # global meta
        student.connect_to_memories()
        student.connect_to_memories_intrinsic()
        student.set_context(context, generate=False)
        if learning:
            student.create_observations()

        # IPython.embed()
        # utils.logger('1 %s' % refs_parses[0][1][0].print_sentence())
        answers = []
        for referent, parse in refs_parses:
            if turk and extrinsic:
                parse = parse[0]
                # tempcolor = referent[0].color
                # tempclass = referent[0].object_class
                # referent[0].color = random.choice(['RED','GREEN','BLUE','BLACK','WHITE','YELLOW','ORANGE','PURPLE','PINK'])
                # referent[0].object_class = random.choice(['BOX','CYLINDER','SPHERE'])
            construction_name = None
            construction_string = None
            construction_sem = None
            result = ''

            utterance = ('the object ' if turk and extrinsic else '')+parse.print_sentence()
            result += 'Teacher describes the %s as: %s\n' % (referent, utterance)
            # utils.logger('Teacher describes the %s as: %s\n' % (referent, utterance))
            # utils.logger('2 %s' % refs_parses[0][1][0].print_sentence())
            guess = None
            # try:
            if cheating:
                lmk_sem = parse.constituents[1].constituents[1].constituents[1].sempole()
                lmk_apps = lmk_sem.ref_applicabilities(context, context.get_all_potential_referents())
                # for key, value in lmk_apps.items():
                #     lmk_apps[key] = float(value == 1.0)
                # IPython.embed()
            elif extrinsic:
                lmk_apps, result1 = student.baseline_prep(utterance)
                result += result1
                lmk_apps[(None,)] = 0.0
                result += str(lmk_apps)+'\n'
            else:
                # lmk_apps = context.get_all_potential_referent_scores()
                lmk_apps = coll.defaultdict(float)
                lmk_apps[(None,)] = 1.0
                for ref in context.get_all_potential_referents():
                    if ref[0].name == 'table':
                        lmk_apps[ref] = 1.0
            # utils.logger('3 %s' % refs_parses[0][1][0].print_sentence())
            baseline1, result1 = student.baseline1(utterance, lmk_apps)
            result += result1
            result += 'Baseline 1 guesses %s\n' % baseline1
            # result += 'Baseline 2 guesses %s\n' % baseline2
            # result += 'Baseline 3 guesses %s\n' % baseline3
            result += 'Correct answer is  %s\n' % referent
            # result += 'guess == referent: %s\n' % (guess==referent)
            # utils.logger(result)
            # IPython.embed()
            # exit()
            # if referent == baseline1:
            #     result += 'Baseline1 is correct.\n'

            if learning:
                utt_keys = student.create_utterance_memories(utterance, referent, lmk_apps=lmk_apps, cheating=cheating)

            # utils.logger('Begin parsing')
            parses = student.parse(utterance)

            # utils.logger('Finished parsing')
            pwidths = sorted([(p.hole_width,p.current[0].count(),p) for p in parses])
            # for width, count, p in pwidths[:10]:
            #     print p.current[0].prettyprint()
            #     print width, count
            #     print
            # raw_input()
            # parses = sorted(parses, key=op.attrgetter('hole_width'))
            # utils.logger('4 %s' % refs_parses[0][1][0].print_sentence())
            _,_,parses = zip(*pwidths)
            if parses[0].hole_width == 0:
                guess = student.choose_referent(parses)
                result += 'Student understands\n'
                result += parses[0].current[0].prettyprint()
                result += 'Student guesses %s\n' % guess
                # utils.logger('Student guesses %s\n' % guess)

                if guess == referent:
                    result += 'Student is correct.\n'
                # utils.logger('5.1 %s' % refs_parses[0][1][0].print_sentence())
            else:
                # except AttributeError, e:
                # result += 'Error: %s\n' % e
                # utils.logger('Error: %s\n' % e)
                # traceback.print_exc()
                # exit()
                result += 'Student could not understand.\n'
                # utils.logger('Student could not understand.')
                result += str(parses[0].parses[0])+'\n'
                result += str(parses[0].current[0].prettyprint())
                completed, result1 = student.construct_from_parses(parses,
                                                                   goal_type)
                result += result1
                if len(completed)==0:
                    result += 'First time student has seen this construction\n'
                    # utils.logger('First time student has seen this construction')
                else:
                    if len(completed) > 1:
                        result += '%s semantically completed\n' % len(completed)
                        # completed = [(parse.equivalence(r),r) for r in completed]
                        # completed.sort(reverse=True)
                        # cwidths = sorted([(c.hole_width,c.current[0].count(),c) for c in completed])
                        # _,_,completed = zip(*cwidths)

                        # complete = completed[0][2]
                        complete = completed[0]
                    else:
                        complete = completed[0]
                        
                    hole = complete.find_partials()[0].get_holes()[0]
                    construction_name = hole.unmatched_pattern.__name__
                    result += '%s\n' % construction_name
                    construction_string = ''.join(hole.collect_leaves())
                    result += '%s\n' % construction_string
                    construction_sem = hole.sempole()
                    result += '%s\n' % construction_sem
                    result += '%s\n' % hole.prettyprint()
                    try:
                        guess, result1 = student.choose_from_tree(complete)
                        result += result1
                    except AssertionError as e:
                        result+=str(e)+'\n'
                        guess = None

                    result += 'Student guesses %s\n' % guess
                    # utils.logger('Student guesses %s\n' % guess)

                    if referent == guess:
                        result += 'Student is correct.\n'

                    # Guess referent
                # utils.logger('5.2 %s' % refs_parses[0][1][0].print_sentence())
                try:
                    if learning:
                        result += 'Creating memories\n'
                        # utils.logger('Creating memories')
                        student.create_new_construction_memories(utt_keys,
                                                                 utterance,
                                                                 parses[:1], 
                                                                 goal_type, 
                                                                 referent)
                        result += 'Done creating memories\n'
                        # utils.logger('Done creating memories')
                except Exception as e:
                    result += 'Exception '+str(e)+'\n'
                    traceback.print_exc()

                # utils.logger('5.3 %s' % refs_parses[0][1][0].print_sentence())


            # if turk and extrinsic:
            #     referent[0].color = tempcolor
            #     referent[0].object_class = tempclass

            assert( construction_name != '' )
            assert( construction_string != '' )
            assert( construction_sem != '' )

            answers.append((time.time(),
                            # construction_name,
                            str(referent),
                            construction_string,
                            construction_sem,
                            referent==guess if guess else None,
                            (referent==baseline1 if baseline1 else None,
                             None,
                             None
                             # referent==baseline2 if baseline2 else None,
                             # referent==baseline3 if baseline3 else None
                             )))
            # result += str(answers)+'\n'
            utils.logger(result)
        student.disconnect()
        # utils.logger('Finishing %s, %s' % (refs_parses[0][0],refs_parses[0][1][0].print_sentence()))
        return answers
    except Exception, exception:
        print 'Heyo!'
        print exception
        traceback.print_exc()
        raise

@automain.automain
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datashelf', type=str)
    parser.add_argument('-n','--num_scenes', type=int, default=None)
    parser.add_argument('-c','--cheating', action='store_true')
    parser.add_argument('-l','--learning', action='store_false')
    args = parser.parse_args()

    student_name = 'Student'

    # meta_grammar = {li.Noun:constraint.ConstraintSet,
    #                 li.Adjective:constraint.ConstraintSet,
    #                 cs.Relation:constraint.ConstraintSet}
    meta_grammar = None

    s_lexicon = [
        li._,
        li.the,
          # li.a_article,
          # li.an_article,

        li.objct,
        li.block,
        li.sphere,
        li.cylinder,
        li.table,
          # li.cube_noun,
          # li.ball,
          # li.box,
          # li.rectangle_noun,
          # li.square_noun,
          # li.shape_noun,
          # li.circle_noun,
          # li.prism_noun,
          # li.cuboid_noun,

        li.corner,
        li.edge,
        li.end,
        li.half,
        li.middle,
        li.side,
          # li.center_noun,
          # li.top_noun,

        li.red,
        li.green,
        li.blue,
        li.white,
        li.black,
          # li.purple,
          # li.yellow,
          # li.orange,
          # li.brown_adj,
          # li.colored_adj,
          # li.pink,
          # li.violet_adj,

          # li.rectangular_adj,
          # li.square_adj,
          # li.round_adj,
          # li.shaped_adj,
          # li.circular_adj,

          # li.small_adj,
          # li.short_adj,
          # li.large_adj,

        li.front,
        li.back,
        li.left,
        li.right,
        #   li.near_dir,
        #   li.far_dir,
        #   li.top_dir,
        li.to,
        li.frm,
        li.of,

        # li.far,
        # li.near,
        # li.on,
        #   li.in_,
        #   li.near_rel,
        #   li.at_rel,
        #   li.next_to_rel,
        #   li.behind_rel,
        #   li.in_front_of_rel,
        #   li.towards_rel,
        #   li.close_to_rel,
        #   li.above_rel,
        #   li.away_from_rel,
    ]
    s_structicon = [
        st.OrientationAdjective,
        st.AdjectivePhrase,
        st.TwoAdjectivePhrase,
        st.DegreeAdjectivePhrase,
        st.NounPhrase,
        st.AdjectiveNounPhrase,
        st.MeasurePhrase,
        st.DegreeMeasurePhrase,
        st.PartOfRelation,
        # st.DistanceRelation,
        # st.OrientationRelation,
        st.ReferringExpression,
        st.RelationLandmarkPhrase,
        st.RelationNounPhrase,
        st.ExtrinsicReferringExpression,

        # st.ReferringExpression2,
        # st.ExtrinsicReferringExpression2,
    ]

    student = lu.LanguageUser(name=student_name, lexicon=s_lexicon, 
                              structicon=s_structicon, meta=meta_grammar,
                              remember=args.learning, reset=args.learning)
    # global engine
    # global meta
    # engine = student.engine
    # student.engine = None
    # meta = student.meta
    # student.meta = None


    goal_type = st.ReferringExpression

    f = shelve.open(args.datashelf)
    seed = f['seed']+1
    random.seed(seed)
    np.random.seed(seed)
    # just_shapes = f['just_shapes']
    # just_objects = f['just_objects']
    extrinsic = f['extrinsic']
    global training_data
    training_data = f['training_data']

    turk = f.has_key('turk')
    if turk:
        utils.logger('Turking')
    random.shuffle(training_data)

    test = f.has_key('test_data')
    if test:
        utils.logger('Testing')
        global test_data
        test_data = f['test_data']
        bad_test_data = f['bad_test_data']
        test_args = [(student.copy(),goal_type,args.cheating,False,extrinsic,turk,None,i) for i in range(len(test_data))]

    if args.num_scenes is not None:
        training_data = training_data[:args.num_scenes]

    utils.logger('Done loading!')
    training_args = [(student.copy(),goal_type,args.cheating,args.learning,extrinsic,turk,i,None) for i in range(len(training_data))]
    poolsize = 7
    pool = mp.Pool(poolsize)
    all_answers = []
    # test_answers = []
    filename = args.datashelf[:-6]+('_cheating' if args.cheating else '') + '_results.shelf'

    epoch_size = 7
    for epoch_training_args in utils.chunks(training_args, epoch_size):
            utils.logger(epoch_training_args)
            all_answers.extend(pool.map(one_scene, epoch_training_args))
            utils.logger(all_answers[-10:])
            f = shelve.open(filename)
            f['all_answers'] = all_answers
            f.close()
        # if test:
        #     test_answers.append([])
        #     for ta in utils.chunks(test_args, poolsize):
        #         test_answers[-1].extend(pool.map(one_scene, ta))
        #         # all_answers.extend(map(one_scene, ta))
        #         utils.logger(test_answers[-10:])
        #         f = shelve.open(filename)
        #         f['test_answers'] = test_answers
        #         f['bad_test_data'] = bad_test_data
        #         f.close()
    exit()
