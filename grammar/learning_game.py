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




def one_scene(args):
    extrinsic = False
    teacher, student, just_objects, just_shapes, goal_type, goal_sem = args
    scene, speaker = sem.run.construct_training_scene(random=True, 
                                                      just_shapes=just_shapes)
    context = cmn.Context(scene, speaker)
    teacher.set_context(context)
    student.connect_to_memories()
    student.set_context(context)


    # conn = engine.connect()
    # scene_key = conn.execute(scenes.insert()).inserted_primary_key[0]

    if just_objects:
        potential_referents = context.get_potential_referents()
    else:
        potential_referents = context.get_all_potential_referents()

    for referent in potential_referents:
        result = ''
        if extrinsic:
            the_parses = teacher.cThe_object__parses
        else:
            the_parses = teacher.landmark_parses

        parse_weights = teacher.weight_parses(referent, 
                                              the_parses)
        parse = teacher.choose_top_parse(parse_weights)

        utterance = parse.print_sentence()
        result += 'Teacher describes the %s as: %s\n' % (referent, utterance)

        guess = None
        try:
            guess = student.choose_referent(utterance)
            result += 'Student guesses %s\n' % guess
            if guess == referent:
                result += 'Student is correct.\n'
        except AttributeError:
            parses = student.parse(utterance)
            parses = sorted(parses, key=op.attrgetter('hole_width'))

            result += 'Student could not understand.\n'
            result += student.construct_from_parses(parses,goal_type,goal_sem)

            # Guess referent

            result += student.create_new_construction_memories(parses, goal_type, 
                                                               referent)


                # result += parse.current[0].prettyprint()+'\n'
            # parse = parses[0]
            # result += 'Best partial parse:\nnum_holes: '+\
            #     '%s, hole_width: %s\n%s' % (parse.num_holes, parse.hole_width, 
            #                                 parse.current[0].prettyprint())
            # result += student.create_relation_memories(parse=parse, 
            #                                            true_referent=referent)
            # result += student.create_relation_construction(parse=parse)

        utils.logger(result)

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
        li.black,
        li.white,
        li.gray,
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
        # li.block,
        # li.box,
        # li.sphere,
        # li.ball,
        # li.cylinder,
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
        li.black,
        li.white,
        li.gray,
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
        st.DistanceRelation,
        st.OrientationRelation,
        st.PartOfRelation,
        st.ReferringExpression,
        st.RelationLandmarkPhrase,
        st.RelationNounPhrase,
        st.ExtrinsicReferringExpression
    ]

    student = lu.LanguageUser(name=student_name, lexicon=s_lexicon, 
                              structicon=s_structicon, meta=meta_grammar,
                              remember=True, reset=True)

    goal_type = st.ReferringExpression
    goal_sem = constraint.ConstraintSet

    utils.logger('Done loading!')
    just_objects=True
    just_shapes=True
    args = [(teacher.copy(), student.copy(),
             just_objects,just_shapes,
             goal_type, goal_sem) 
            for _ in range(5)]
    # pool = mp.Pool(7)
    # pool.map(one_scene, args)
    map(one_scene, args)

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
