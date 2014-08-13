import sys
sys.path.insert(1,'..')
import automain
import argparse
import random
import shelve
import traceback
import random
import functools as ft
import multiprocessing as mp
import sqlalchemy as alc

import utils
import common as cmn
import semantics as sem
import language_user as lu
import lexical_items as li
import constructions as st
import partial_parser as pp
from all_parse_generator import AllParseGenerator as apg

import IPython

f = shelve.open('test.shelf')
training_data = f['training_data']
f.close()

student_name = 'Student'

# meta_grammar = {li.Noun:constraint.ConstraintSet,
#                 li.Adjective:constraint.ConstraintSet,
#                 cs.Relation:constraint.ConstraintSet}

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
    # li.far,
    # li.near,
    # li.on,

    li.a_article,
    li.an_article,

    # li.top_noun,
    # li.bottom_noun,
    # li.back_noun,
    # li.rear_noun,
    # li.front_noun,
    li.center_noun,
    li.shape_noun,
    li.circle_noun,
    li.prism_noun,
    li.cube_noun,
    li.cuboid_noun,
    # li.disc_noun,

    # li.lower_adj,
    # li.upper_adj,
    li.near_adj,
    li.far_adj,
    li.top_adj,
    # li.bottom_adj,
    # li.cylindrical_adj,
    # li.circular_adj,
    li.rectangular_adj,
    li.square_adj,
    li.coloured_adj,
    li.brown_adj,
    # li.violet_adj,
    # li.black_adj,

    # li.big_adj,
    # li.large_adj,
    li.small_adj,
    # li.tall_adj,
    # li.short_adj,
    # li.wide_adj,

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
                          structicon=s_structicon, meta=None,
                          remember=True, reset=True)

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
    li.on,

    li.a_article,
    li.an_article,

    # li.top_noun,
    # li.bottom_noun,
    # li.back_noun,
    # li.rear_noun,
    # li.front_noun,
    li.center_noun,
    li.shape_noun,
    li.circle_noun,
    li.prism_noun,
    li.cube_noun,
    li.cuboid_noun,
    # li.disc_noun,

    # li.lower_adj,
    # li.upper_adj,
    li.near_adj,
    li.far_adj,
    li.top_adj,
    # li.bottom_adj,
    # li.cylindrical_adj,
    # li.circular_adj,
    li.rectangular_adj,
    li.square_adj,
    li.coloured_adj,
    li.brown_adj,
    # li.violet_adj,
    # li.black_adj,

    # li.big_adj,
    # li.large_adj,
    li.small_adj,
    # li.tall_adj,
    # li.short_adj,
    # li.wide_adj,

    li.at_rel,
    li.next_to_rel,
    li.near_rel,
    li.towards_rel,
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
    st.ReferringExpression,
    st.RelationLandmarkPhrase,
    st.RelationNounPhrase,
    st.ExtrinsicReferringExpression,

    # st.ReferringExpression2,
    # st.ExtrinsicReferringExpression2,
]

teacher = lu.LanguageUser(name='Teacher', lexicon=t_lexicon, 
                          structicon=t_structicon, meta=None,
                          remember=False)

# all_parses = apg.generate_parses(st.ReferringExpression,
#                                  s_lexicon,
#                                  s_structicon)

# a = all_parses[-1]

engine = alc.create_engine('sqlite:///mtbolt.db')
meta = alc.MetaData()
meta.reflect(bind=engine)
entities = meta.tables['scenes_entity']
descriptions = meta.tables['tasks_descriptionquestion']

# q = alc.sql.select([entities.c.scene_id, 
#                     entities.c.name, 
#                     descriptions.c.object_description]).where(
#                     entities.c.id==descriptions.c.entity_id)
q = alc.sql.select([entities.c.scene_id, 
                    entities.c.name, 
                    descriptions.c.location_description]).where(
                    entities.c.id==descriptions.c.entity_id)
conn = engine.connect()
results = [x for x in list(conn.execute(q)) if len(x[2])>0]
# results = [x for x in zip(*list(conn.execute(q)))[0] if len(x)>0]
conn.close()
scene_ids, entity_names, descriptions = zip(*results)
IPython.embed()
exit()

scene_infos = sem.run.read_scenes('static_scenes',True)
scenes = {}
for num, scene, speaker in scene_infos:
    scenes[num] = (scene, speaker)



def strip_string(string, to_strip):
    if to_strip:
        while string.startswith(to_strip):
            string = string[len(to_strip):]
        while string.endswith(to_strip):
            string = string[:-len(to_strip)]
    return string

def parse(utterance):
    parses = list(teacher.parse(utterance, max_holes=0))
    return parses

# pool = mp.Pool(7)
import time
t = time.time()
parses = map(parse,descriptions)
print time.time()-t
pairs = [(scenes[s], (scenes[s][0].landmarks['object_'+e],),d,p[0]) 
         for s,e,d,p in zip(scene_ids,entity_names,descriptions,parses) 
         if len(p) == 1]
# refex_pairs = [(s[0],s[1],((e,p.current[0]),)) for s,e,r,p in pairs if isinstance(p.current[0],st.ReferringExpression)]
# print len(refex_pairs)
relations_pairs = [(s[0],s[1],((e,p.current[0]),)) for s,e,r,p in pairs if isinstance(p.current[0],st.RelationLandmarkPhrase)]
print len(relations_pairs)

# import collections as coll
# c = coll.Counter()
# import re
# [c.update(re.split(' |,|, |;|; ',u.lower().strip().strip('.'))) for u in results]

name = 'turk_relation_training.shelf'
f = shelve.open(name)
f['turk'] = True
f['seed'] = 0
f['extrinsic'] = True
f['training_data'] = relations_pairs
f.close()

# IPython.embed()