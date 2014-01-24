#!/usr/bin/python
import sys
sys.path.insert(1,'..')
import semantics as sem

import automain
import utils
import random
import common as cmn
import language_user as lu
import lexical_items as li
import constructions as st

import operator as op
import matplotlib as mpl

@automain.automain
def main():

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

    teacher = lu.LanguageUser('Teach', lexicon=t_lexicon, 
                              structicon=t_structicon, remember=False)

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

    student = lu.LanguageUser('Stud', lexicon=s_lexicon, 
                              structicon=s_structicon, remember=False)

    scene_descs=sem.run.read_scenes('static_scenes/',normalize=True,image=True)

    # mpl.pyplot.ion()
    for scene, speaker, image in scene_descs:
        # mpl.pyplot.imshow(image)
        # mpl.pyplot.show()
        # scene, speaker = sem.run.construct_training_scene(random=True)

        context = cmn.Context(scene, speaker)
        teacher.set_context(context)
        student.set_context(context)

        utils.logger(scene)
        for referent in context.get_all_potential_referents():
        # referent = random.choice(context.get_potential_referents())
            utils.logger(referent)
            parse = teacher.choose_top_parse(
                        teacher.weight_parses(
                            referent, teacher.landmark_parses))

            utterance = parse.print_sentence()

            utils.logger('Teacher describes the %s as: %s' % (referent, utterance))

            try:
                guess = student.choose_referent(utterance)
                utils.logger('Student guesses %s' % guess)
            except AttributeError:
                parses = student.parse(utterance)
                parses = sorted(parses, key=op.attrgetter('hole_width'))

                utils.logger('Student could not understand.')
                parse = parses[0]
                utils.logger('Best partial parse:\nnum_holes: '
                             '%s, hole_width: %s\n%s' % 
                             (parse.num_holes, parse.hole_width, 
                              parse.current[0].prettyprint()))
                for hole in parse.current[0].find_holes():
                    utils.logger(hole.prettyprint())
