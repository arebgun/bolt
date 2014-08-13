#!/usr/bin/python
import sys
sys.path.insert(1,'..')
import automain
import argparse
import random
# import shelve
import traceback
import multiprocessing as mp

import utils
import common as cmn
import semantics as sem
import language_user as lu
import lexical_items as li
import constructions as st

import lockshelf

global count
count = []

def fake_one_scene(args):
    teacher,just_shapes,just_objects,extrinsic, i = args
    # global count
    # # count += 1; print count
    # count.append(i)
    print i#, len(count)
    scene, speaker = sem.run.construct_training_scene(random=True, 
                            just_shapes=just_shapes)

def one_scene(args):
    try:
        # global training_data
        teacher,just_shapes,just_objects,extrinsic,i,filename = args
        # global count
        # # count += 1; print count
        # count.append(i)
        print i#, len(count)
        scene, speaker = sem.run.construct_training_scene(random=True, 
                                just_shapes=just_shapes)

        context = cmn.Context(scene, speaker)
        teacher.set_context(context)

        if just_objects:
            potential_referents = context.get_potential_referents()
        else:
            potential_referents = context.get_all_potential_referents()

        if extrinsic:
            the_parses = teacher.The_object__parses
        else:
            the_parses = teacher.landmark_parses

        refs_parses = []
        for referent in potential_referents:
            # sys.stdout.write('.')
            parse_weights = teacher.weight_parses(referent, the_parses)
            # utils.logger(referent)
            # for score, parse in parse_weights[:20]:
            #     utils.logger((score, parse.print_sentence()))
            # utils.logger('')
            # utils.logger('')
            parse = teacher.choose_top_parse(parse_weights)
            utils.logger(parse.print_sentence())
            refs_parses.append((referent, parse))
            # utterance = parse.print_sentence()
        print
        # import pickle
        # temp = pickle.loads(pickle.dumps((scene,None,refs_parses)))

        x = (scene,speaker,refs_parses)
        written = False
        while not written:
            try:
                f = lockshelf.open(filename)
                training_data = f['training_data']
                training_data.append(x)
                f['training_data'] = training_data
                utils.logger(len(f['training_data']))
                f.close()
                written = True
            except IOError as e:
                utils.logger(e)

        return

    except Exception, exception:
        print 'Heyo!'
        print exception
        traceback.print_exc()
        raise


@automain.automain
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    # parser.add_argument('num_scenes', type=int)
    parser.add_argument('-a', '--append', action='store_true')
    args = parser.parse_args()

    teacher_name = 'Teacher'

    random.seed(args.seed)

    num_scenes = 200
    name = '2rel_extrinsic_training'
    just_shapes = False
    just_objects = True
    extrinsic = True
    # name = 'noshapes'

    # t_lexicon = [
    #     li._,
    #     li.the,
    #     li.objct,
    #     li.block,
    #     # li.box,
    #     li.sphere,
    #     # li.ball,
    #     li.cylinder,
    #     li.table,
    #     li.corner,
    #     li.edge,
    #     li.end,
    #     li.half,
    #     li.middle,
    #     li.side,
    #     li.red,
    #     # li.orange,
    #     # li.yellow,
    #     li.green,
    #     li.blue,
    #     # li.purple,
    #     # li.pink,
    #     li.black,
    #     li.white,
    #     # li.gray,
    #     li.front,
    #     li.back,
    #     li.left,
    #     li.right,
    #     li.to,
    #     li.frm,
    #     li.of,
    #     li.far,
    #     li.near,
    #     li.on
    # ]

    extended_t_lexicon = [
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
          # li.near_dir,
          # li.far_dir,
          # li.top_dir,
        li.to,
        li.frm,
        li.of,

        li.far,
        li.near,
        li.on,
          # li.in_,
          # li.near_rel,
          # li.at_rel,
          # li.next_to_rel,
          # li.behind_rel,
          # li.in_front_of_rel,
          # li.towards_rel,
          # li.close_to_rel,
          # li.above_rel,
          # li.away_from_rel,
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
        st.DistanceRelation,
        st.OrientationRelation,
        st.PartOfRelation,
        st.ReferringExpression,
        st.RelationLandmarkPhrase,
        st.RelationNounPhrase,
        st.ExtrinsicReferringExpression
    ]

    teacher = lu.LanguageUser(name=teacher_name, lexicon=extended_t_lexicon, 
                              structicon=t_structicon, meta=None,
                              remember=False)


    pool = mp.Pool(8)
    # training_data = pool.map(one_scene, gen_args)


    filename = '%s_seed%i_num%i' % (name, args.seed, num_scenes)
    if just_shapes:
        filename += '_js'
    if just_objects:
        filename += '_jo'
    if extrinsic:
        filename += '_e'
    # time.asctime(time.localtime()).replace(' ','_').replace(':',''))
    filename += '.shelf'
    print filename

    gen_args = [(teacher.copy(),just_shapes,just_objects,extrinsic,i,filename) for i in range(num_scenes)]
    # poolsize = 7
    # pool = mp.Pool(poolsize)

    # if args.append:
    #     training_data = map(fake_one_scene, gen_args)
    #     chunk_size = 8
    #     for ga in utils.chunks(gen_args, chunk_size):
    #         training_data = map(one_scene, ga)
    #         f = shelve.open(filename)
    #         f['training_data'] = f['training_data'] + training_data
    #         f.close()
    # else:
    # training_data = map(one_scene, gen_args)
    training_data = []
    f = lockshelf.open(filename)
    f['seed'] = args.seed
    f['just_shapes'] = just_shapes
    f['just_objects'] = just_objects
    f['extrinsic'] = extrinsic
    f['training_data'] = training_data
    f.close()
    for series in utils.chunks(gen_args, 10):
        training_data.extend(pool.map(one_scene, series))

