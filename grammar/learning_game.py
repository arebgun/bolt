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
import gen2_features as g2f

import os

import matplotlib.pyplot as plt

import IPython
import traceback
import shelve
import time


# global answers
# answers = []

def one_scene(args):
    # global answers
    try:
        teacher,student,just_objects,just_shapes,extrinsic,goal_type = args
        scene, speaker = sem.run.construct_training_scene(random=True, 
                                                          just_shapes=just_shapes)
        utils.logger(scene)
        context = cmn.Context(scene, speaker)
        teacher.set_context(context)
        student.connect_to_memories()
        student.set_context(context)

        # IPython.embed()

        if just_objects:
            potential_referents = context.get_potential_referents()
        else:
            potential_referents = context.get_all_potential_referents()

        answers = []
        for referent in potential_referents:
            construction_name = None
            construction_string = None
            construction_sem = None
            result = ''
            if extrinsic:
                the_parses = teacher.The_object__parses
            else:
                the_parses = teacher.landmark_parses
            parse_weights = teacher.weight_parses(referent, 
                                                  the_parses)
            parse = teacher.choose_top_parse(parse_weights)

            utterance = parse.print_sentence()
            result += 'Teacher describes the %s as: %s\n' % (referent, utterance)
            # utils.logger('Teacher describes the %s as: %s\n' % (referent, utterance))
            result += str(parse.sempole()
                               .ref_applicabilities(context,
                                                    potential_referents))+'\n'

            guess = None
            try:
                # utils.logger('Begin parsing')
                parses = student.parse(utterance)

                # utils.logger('Finished parsing')
                parses = sorted(parses, key=op.attrgetter('hole_width'))
                guess = student.choose_referent(parses)
                result += 'Student guesses %s\n' % guess
                # utils.logger('Student guesses %s\n' % guess)
                result += str(parses[0].current[0].sempole()
                                .ref_applicabilities(context,
                                                    potential_referents))+'\n'
                if guess == referent:
                    result += 'Student is correct.\n'
            except AttributeError, e:
                # print e
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
                        # utils.logger(parse)
                        # utils.logger(parse.prettyprint())
                        completed = [(parse.equivalence(r),r) for r in completed]
                        completed.sort(reverse=True)
                        # completed = [(parse.get_hole_size(),r) for r in completed]
                        # completed.sort()
                        # for e,c in completed[:5]:
                        #     result += '%s' % c.prettyprint()
                        #     result += '%s\n' % e
                        complete = completed[0][1]
                    else:
                        complete = completed[0]
                        
                    # result += '%s\n' % complete.prettyprint()
                    # result += '%s\n\n' % complete.sempole()
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
                    result += str(complete.sempole()
                               .ref_applicabilities(context,
                                                    potential_referents))+'\n'
                    if referent == guess:
                        result += 'Student is correct.\n'

                # Guess referent

                try:
                    # utils.logger('Creating memories')
                    student.create_new_construction_memories(utterance,
                                                             parses, 
                                                             goal_type, 
                                                             referent)
                    # utils.logger('Done creating memories')
                except Exception as e:
                    result += str(e)+'\n'

            answers.append((time.time(),
                            construction_name,
                            construction_string,
                            construction_sem,
                            referent==guess if guess!=None else None))
            # result += str(answers)+'\n'
            utils.logger(result)
    except Exception, exception:
        print exception
        traceback.print_exc()
        raise
    return answers

@automain.automain
def main():

    teacher_name = 'Teacher'
    student_name = 'Student'
    # db_suffix = '_memories.db'
    # if os.path.isfile(teacher_name + db_suffix):
    #     os.remove(teacher_name + db_suffix)

    meta_grammar = {li.Noun:constraint.ConstraintSet,
                    li.Adjective:constraint.ConstraintSet,
                    cs.Relation:constraint.ConstraintSet}

    t_lexicon = [
        li._,
        li.the,
        li.objct,
        li.block,
        # li.box,
        li.sphere,
        # li.ball,
        li.cylinder,
        li.table,
        li.corner,
        li.edge,
        li.end,
        li.half,
        li.middle,
        li.side,
        li.red,
        # li.orange,
        # li.yellow,
        li.green,
        li.blue,
        # li.purple,
        # li.pink,
        li.black,
        li.white,
        # li.gray,
        li.front,
        li.back,
        li.left,
        li.right,
        li.to,
        li.frm,
        li.of,
        li.far,
        li.near,
    ]
    t_structicon = [
        st.OrientationAdjective,
        st.AdjectivePhrase,
        st.TwoAdjectivePhrase,
        st.DegreeAdjectivePhrase,
        st.NounPhrase,
        st.AdjectiveNounPhrase,
        st.MeasurePhrase,
        st.DegreeMeasurePhrase,
        st.PartOfRelation,
        st.DistanceRelation,
        st.OrientationRelation,
        st.PartOfRelation,
        st.ReferringExpression,
        st.RelationLandmarkPhrase,
        st.RelationNounPhrase,
        st.ExtrinsicReferringExpression
    ]

    teacher = lu.LanguageUser(name=teacher_name, lexicon=t_lexicon, 
                              structicon=t_structicon, meta=meta_grammar,
                              remember=False)

    s_lexicon = [
        li._,
        li.the,
        li.objct,
        li.block,
        # li.box,
        li.sphere,
        # li.ball,
        li.cylinder,
        li.table,
        li.corner,
        li.edge,
        li.end,
        li.half,
        li.middle,
        li.side,
        # li.red,
        # li.orange,
        # li.yellow,
        # li.green,
        # li.blue,
        # li.purple,
        # li.pink,
        # li.black,
        # li.white,
        # li.gray,
        li.front,
        li.back,
        li.left,
        li.right,
        li.to,
        li.frm,
        li.of,
        li.far,
        li.near,
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
        st.ExtrinsicReferringExpression
    ]

    student = lu.LanguageUser(name=student_name, lexicon=s_lexicon, 
                              structicon=s_structicon, meta=meta_grammar,
                              remember=True, reset=True)
    # student.connect_to_memories()
    # student.construct('Relation','near to')
    # exit()
    goal_type = st.ReferringExpression

    utils.logger('Done loading!')
    just_objects=True
    just_shapes=False
    extrinsic=False
    num_scenes = 64
    args = [(teacher.copy(), student.copy(),
             just_objects,just_shapes,extrinsic,
             goal_type) 
            for _ in range(num_scenes)]
    pool = mp.Pool(8)
    all_answers = pool.map(one_scene, args)
    # all_answers = map(one_scene, args)
    utils.logger(all_answers)

    filename = 'learning_game_results.shelf'
    f = shelve.open(filename)
    f['all_answers'] = all_answers
    f.close()
    exit()


    # scene_descs=sem.run.read_scenes('static_scenes/',normalize=True,image=True)

    # # mpl.pyplot.ion()
    # for scene, speaker, image in scene_descs:
    #     # mpl.pyplot.imshow(image)
    #     # mpl.pyplot.show()

    #     context = cmn.Context(scene, speaker)
    #     teacher.set_context(context)
    #     student.set_context(context)

    #     utils.logger(scene)
    #     for referent in context.get_potential_referents():
    #     # referent = random.choice(context.get_potential_referents())
    #     # utils.logger(referent)
    #         parse = teacher.choose_top_parse(
    #                     teacher.weight_parses(
    #                         referent, teacher.The_object__parses))

    #         utterance = parse.print_sentence()

    #         utils.logger('Teacher describes the %s as: %s' % (referent, utterance))

    #         try:
    #             guess = student.choose_referent(utterance)
    #             utils.logger('Student guesses %s' % guess)
    #         except AttributeError:
    #             parses = student.parse(utterance)
    #             parses = sorted(parses, key=op.attrgetter('hole_width'))

    #             utils.logger('Student could not understand.')
    #             parse = parses[0]
    #             utils.logger('Best partial parse:\nnum_holes: '
    #                          '%s, hole_width: %s\n%s' % 
    #                          (parse.num_holes, parse.hole_width, 
    #                           parse.current[0].prettyprint()))
    #             for hole in parse.current[0].find_holes():
    #                 utils.logger(hole.prettyprint())
