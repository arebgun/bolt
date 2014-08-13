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

global training_data
# global answers
# answers = []


def one_scene(args):
    # global answers
    try:
        student, goal_type, cheating, extrinsic, turk, i = args

        global training_data
        scene, speaker, refs_parses = training_data[i]
        utils.logger(scene)
        context = cmn.Context(scene, speaker)
        # global engine
        # global meta
        student.connect_to_memories()
        student.connect_to_memories_intrinsic()
        student.set_context(context)
        student.create_observations()

        # IPython.embed()

        answers = []
        for referent, parse in refs_parses:
            construction_name = None
            construction_string = None
            construction_sem = None
            result = ''

            utterance = ('the object ' if turk else '')+parse.print_sentence()
            result += 'Teacher describes the %s as: %s\n' % (referent, utterance)
            # utils.logger('Teacher describes the %s as: %s\n' % (referent, utterance))

            guess = None
            # try:
            if cheating:
                lmk_sem = parse.constituents[1].constituents[1].constituents[1].sempole()
                lmk_apps = lmk_sem.ref_applicabilities(context, context.get_all_potential_referents())
                # for key, value in lmk_apps.items():
                #     lmk_apps[key] = float(value == 1.0)
                # IPython.embed()
            elif extrinsic:
                lmk_apps = student.baseline_prep(utterance)
                lmk_apps[(None,)] = 0.0
                result += str(lmk_apps)+'\n'
            else:
                lmk_apps = context.get_all_potential_referent_scores()
                lmk_apps[(None,)] = 1.0

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
            _,_,parses = zip(*pwidths)
            if parses[0].hole_width == 0:
                guess = student.choose_referent(parses)
                result += 'Student understands\n'
                result += parses[0].current[0].prettyprint()
                result += 'Student guesses %s\n' % guess
                # utils.logger('Student guesses %s\n' % guess)

                if guess == referent:
                    result += 'Student is correct.\n'
            else:
                # except AttributeError, e:
                # result += 'Error: %s\n' % e
                # utils.logger('Error: %s\n' % e)
                # traceback.print_exc()
                # exit()
                result += 'Student could not understand.\n'
                # utils.logger('Student could not understand.')
                completed, result1 = student.construct_from_parses(parses,
                                                                   goal_type)
                # result += result1
                if len(completed)==0:
                    result += 'First time student has seen this construction\n'
                    # utils.logger('First time student has seen this construction')
                else:
                    if len(completed) > 1:
                        completed = [(parse.equivalence(r),r) for r in completed]
                        completed.sort(reverse=True)

                        complete = completed[0][1]
                    else:
                        complete = completed[0]
                        
                    hole = complete.find_partials()[0].get_holes()[0]
                    construction_name = hole.unmatched_pattern.__name__
                    result += '%s\n' % construction_name
                    construction_string = hole.print_sentence()
                    result += '%s\n' % construction_string
                    construction_sem = hole.sempole()
                    result += '%s\n' % construction_sem
                    result += '%s\n' % hole.prettyprint()
                    result += '%s\n\n' % hole.sempole()
                    guess = student.choose_from_tree(complete)

                    result += 'Student guesses %s\n' % guess
                    # utils.logger('Student guesses %s\n' % guess)

                    if referent == guess:
                        result += 'Student is correct.\n'

                    # Guess referent

                try:
                    utils.logger('Creating memories')
                    student.create_new_construction_memories(utt_keys,
                                                             utterance,
                                                             parses, 
                                                             goal_type, 
                                                             referent)
                    utils.logger('Done creating memories')
                except Exception as e:
                    result += str(e)+'\n'




            assert( construction_name != '' )
            assert( construction_string != '' )
            assert( construction_sem != '' )

            answers.append((time.time(),
                            construction_name,
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
    except Exception, exception:
        print 'Heyo!'
        print exception
        traceback.print_exc()
        raise
    student.disconnect()
    return answers

@automain.automain
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datashelf', type=str)
    parser.add_argument('-n','--num_scenes', type=int, default=None)
    parser.add_argument('-c','--cheating', action='store_true')
    args = parser.parse_args()

    student_name = 'Student'

    meta_grammar = {li.Noun:constraint.ConstraintSet,
                    li.Adjective:constraint.ConstraintSet,
                    cs.Relation:constraint.ConstraintSet}

    s_lexicon = [
        li._,
        li.the,
        li.objct,
        li.block,
        li.box,
        li.sphere,
        li.ball,
        li.cylinder,
        li.table,
        li.corner,
        li.edge,
        li.end,
        li.half,
        li.middle,
        li.side,
        li.red,
        li.orange,
        li.yellow,
        li.green,
        li.blue,
        li.purple,
        li.pink,
        li.front,
        li.back,
        li.left,
        li.right,
        li.to,
        li.frm,
        li.of,
        # li.far,
        # li.near,
        # li.on,

        li.a_article,
        li.an_article,

        li.center_noun,
        li.shape_noun,
        li.circle_noun,
        li.prism_noun,
        li.cube_noun,
        li.cuboid_noun,

        li.near_adj,
        li.far_adj,
        li.top_adj,
        li.rectangular_adj,
        li.square_adj,
        li.coloured_adj,
        li.brown_adj,

        li.small_adj,

        # li.at_rel,
        # li.next_to_rel,
        # li.near_rel,
        # li.towards_rel,
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
                              remember=True, reset=True)
    # global engine
    # global meta
    # engine = student.engine
    # student.engine = None
    # meta = student.meta
    # student.meta = None


    goal_type = st.ReferringExpression

    f = shelve.open(args.datashelf)
    seed = f['seed']
    random.seed(seed)
    np.random.seed(seed)
    # just_shapes = f['just_shapes']
    # just_objects = f['just_objects']
    extrinsic = f['extrinsic']
    global training_data
    training_data = f['training_data']
    turk = f.has_key('turk')
    if turk:
        random.shuffle(training_data)
    if args.num_scenes is not None:
        training_data = training_data[:args.num_scenes]

    utils.logger('Done loading!')
    training_args = [(student.copy(),goal_type,args.cheating,extrinsic,turk,i) for i in range(len(training_data))]
    poolsize = 8
    pool = mp.Pool(poolsize)
    all_answers = []
    filename = args.datashelf[:-6]+('_cheating' if args.cheating else '') + '_results.shelf'
    for ta in utils.chunks(training_args, poolsize):
        all_answers.extend(pool.map(one_scene, ta))
        # all_answers.extend(map(one_scene, ta))
        utils.logger(all_answers)
        f = shelve.open(filename)
        f['all_answers'] = all_answers
        f.close()

    exit()
