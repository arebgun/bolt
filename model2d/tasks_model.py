#!/usr/bin/env python
# coding: utf-8

from __future__ import division
import os
import sys
sys.path.insert(1,"..")


from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.ext.declarative import _declarative_constructor
from sqlalchemy import func

from utils import force_unicode, bigrams, trigrams, lmk_id, logger

import numpy as np
from collections import defaultdict

from semantics import run

from parse import parse_sentences, modify_parses, get_modparse, ParseError
from models import SentenceParse

import object_correction_testing
import testing_testing

from location_from_sentence import get_all_sentence_posteriors

import tempfile
import subprocess
import re
import utils
from planar import Vec2
print Vec2.almost_equals_points
from utils import m2s
from itertools import izip, product

import shelve

### configuration ###

db_url = 'sqlite:///mtbolt.db'
echo = False

### setting up sqlalchemy stuff ###

engine = create_engine(db_url, echo=echo)
Session = sessionmaker()
Session.configure(bind=engine)
session = Session()


# as seen on http://stackoverflow.com/a/1383402
class ClassProperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

class Base(object):
    # if you have a __unicode__ method
    # you get __str__ and __repr__ for free!
    def __str__(self):
        return unicode(self).encode('utf-8')

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self)

    # figure out your own table name
    @declared_attr
    def __tablename__(cls):
        # table names should be plural
        return cls.__name__.lower() + 's'

    # easy access to `Query` object
    #@ClassProperty
    @classmethod
    def query(cls, *args, **kwargs):
        return session.query(cls, *args)

    # like in elixir
    @classmethod
    def get_by(cls, **kwargs):
        return cls.query().filter_by(**kwargs).first()

    # like in django
    @classmethod
    def get_or_create(cls, defaults={}, **kwargs):
        obj = cls.get_by(**kwargs)
        if not obj:
            kwargs.update(defaults)
            obj = cls(**kwargs)
        return obj

    def _constructor(self, **kwargs):
        _declarative_constructor(self, **kwargs)
        # add self to session
        session.add(self)

Base = declarative_base(cls=Base, constructor=Base._constructor)

def create_all():
    """create all tables"""
    Base.metadata.create_all(engine)


class scenes_entity(Base):
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    id = Column(Integer, primary_key=True)
    created = Column(DateTime, nullable=False)
    modified = Column(DateTime, nullable=False)
    scene_id = Column(Integer)
    name = Column(String)



class scenes_scene(Base):
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    id = Column(Integer, primary_key=True)
    created = Column(DateTime, nullable=False)
    modified = Column(DateTime, nullable=False)
    name = Column(String)
    image = Column(String)


class tasks_descriptionquestion(Base):
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    id = Column(Integer, primary_key=True)

    created = Column(DateTime, nullable=False)
    modified = Column(DateTime, nullable=False)

    task_id = Column(Integer)
    scene_id = Column(Integer)
    entity_id = Column(Integer)

    answer = Column(String)
    object_description = Column(String)
    location_description = Column(String)
    # use_in_object_tasks = Column(Boolean)

class object_tasks_entitybinding(Base):
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    id = Column(Integer, primary_key=True)

    created = Column(DateTime, nullable=False)
    modified = Column(DateTime, nullable=False)

    task_id = Column(Integer)
    description_id = Column(Integer)
    binding = Column(Integer)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-iterations', type=int, default=1)
    parser.add_argument('-u', '--update-scale', type=int, default=1000)
    parser.add_argument('-p', '--num-processors', type=int, default=7)
    parser.add_argument('-s', '--num-samples', action='store_true')
    parser.add_argument('-d', '--scene-directory', type=str)
    parser.add_argument('-t', '--tag', type=str)

    args = parser.parse_args()

    engine.echo = False
    create_all()

    # f = shelve.open('processed_data.shelf')
    # if not f.has_key('all_scenes'):

    # test_indices = [3411,2701,1764,264,1142,852,2028,2774,3341,161,779,536,3853,357,249,2569,4175,2971,1368,3305,2586,591,3710,1909,
    # 4085,443,582,3895,3291,3535,2487,3204,476,3042,596,3524,1206,1284,4293,1168,410,3417,1332,892,623,2406,778,3472,2864,4174,1462,
    # 3416,453,4397,2036,1017,1033,3491,1207,4163,1092,2897,1445,2990,473,1155,510,1671,3957,561,1043,182,1854,238,2900,444,987,3041,
    # 612,1167,3594,2739,677,2585,2170,636,3940,2680,848,2938,2328,2829,3331,2871,2122,2169,3093,884,3557,525,3684,2277,4399,1135,1703,
    # 3268,3679,686,2448,992,3655,3711,1748,3881,3473,1756,3518,2376,3223,265,3026,3516,3749,1893,1496,4065,1280,378,3025,3348,820,1585,
    # 2257,3146,3147,3094,3198,3064,4189,693,1707,3831,3606,2758,988,4126,4352,2218,883,3288,2620,2227,2745,2775,4405,1061,2907,2753,3027,
    # 370,3904,2401,3555,978,1958,1930,3038,2439,962,491,4317,2314,2334,2638,3312,3724,3455,3435,543,2444,3997,4049,3935,3842,1527,735,4053,
    # 4135,3539,730,2482,3188,825,3037,4395,3657,428,3205,1490,3805]

    test_set = defaultdict(list)
    for o in object_tasks_entitybinding.query().all():
        test_set[o.description_id].append( o.binding )

    words = {'between':0,
             'among':0,
             'objects':0,
             'furthest':0,
             'farthest':0,
             'highest':0,
             'closest':0,
             'lowest':0,
             'first':0,
             'second':0,
             'third':0,
             'fourth':0,
             'fifth':0,
             'last':0,
             'viewer':0,
             'me':0}
    no_bad_words = [
]
    typos = []

    try:
        from enchant.checker import SpellChecker
        chkr = SpellChecker("en_US")
    except:
        pass
    total = 0

    # run.read_scenes(sys.argv[1])
    all_scenes   = []
    all_descs = []
    turkers_correct = []
    all_ids = []
    for s in scenes_scene.query().all():
        entity_names = []
        lmks         = []
        lmk_descs    = []
        loc_descs    = []
        ids = []
        #print s.id, s.name
        for scene, speaker in run.read_scenes(os.path.join(args.scene_directory,s.name), normalize=True):
            logger( str(scene.landmarks['table'].representation.rect.width) + ', ' + str(scene.landmarks['table'].representation.rect.height))
            for t in tasks_descriptionquestion.query().filter(tasks_descriptionquestion.scene_id==s.id).all():
                #print t.id, t.scene_id, t.entity_id, t.answer, t.object_description, t.location_description
                entity_name = scenes_entity.query().filter(scenes_entity.id==t.entity_id).one().name
                lmk = scene.landmarks['object_%s' % entity_name]
                lmk_desc = t.object_description
                loc_desc = t.location_description
                description_id = t.id

                if description_id in test_set:
                    bindings = []
                    if len(test_set[description_id]) < 5:
                        del test_set[description_id]
                    else:
                        for binding in test_set[description_id]:
                            turkers_correct.append( binding == int(entity_name) )
                            bindings.append( 'object_%s' % binding )
                        test_set[description_id] = bindings

                # print lmk, entity_name, lmk_desc, loc_desc
                if loc_desc:
                    all_descs.append(loc_desc)

                    if not description_id in test_set:
                        if not ('objects' in loc_desc or 'between' in loc_desc):
                            total += 1
                        good = True
                        for word in words.keys():
                            if word in loc_desc:
                                words[word]+=1
                                good = False

                        try:
                            chkr.set_text(loc_desc.lower())
                            for err in chkr:
                                # print "ERROR:", err.word
                                if not ('colour' in err.word or 'centre' in err.word):
                                    typos.append(err.word)
                                    good = False
                        except:
                            pass

                        if good:
                            no_bad_words.append(description_id)

            # chunks = [loc_desc]
                    chunks = []
                    # print loc_desc
                    for part in re.split('\.|;',loc_desc.lower().strip()):
                        if part != u'':
                            for partt in part.strip().split(' and '):
                                partt = re.sub('^.*? is ','',partt)
                                if partt.strip() != u'':
                                    for parttt in partt.strip().split(','):
                                        if parttt.strip() != u'':
                                            all_descs.append(parttt.strip())
                                            chunks.append(parttt.strip())
                    #                         # print '  ',parttt
                    # raw_input()

                    entity_names.append(entity_name)
                    lmks.append(lmk)
                    lmk_descs.append(lmk_desc)
                    # loc_descs.append(loc_desc)
                    # loc_descs.append(parttt.strip())
                    loc_descs.append(chunks)
                    ids.append( description_id )
            all_ids.extend(ids)

            all_scenes.append( {'scene':scene,'speaker':speaker,'lmks':lmks,'loc_descs':loc_descs, 'ids':ids} )


    print 'loaded', len(all_scenes), 'scenes'
    print 'Turkers_correct: Total: %i, Out of: %i, Fraction: %f' % (sum(turkers_correct),len(turkers_correct),float(sum(turkers_correct))/len(turkers_correct))


    #choose new test set
    from random import shuffle
    shuffle(all_ids)
    test_set = all_ids[:350]

    # import shelve
    # f = shelve.open('testing_u1000_not_memory_Fri_Feb__8_145640_2013.shelf')
    # f['turk_answers'] = test_set
    # f.close()
    # exit()

    # for value in test_set.values():
    #     print value
    # exit()

    # print words
    # print 'Sum:',sum(words.values())
    # # print typos
    # print 'Typos:',len(typos)
    # print 'Total:',total
    # print 'Good:',len(no_bad_words)

    # import IPython
    # IPython.embed()
    # exit()


    sp_db = SentenceParse.get_sentence_parse(all_descs[0])
    try:
        res = sp_db.all()[0]
    except IndexError:

        parses = parse_sentences(all_descs,n=5,threads=8)

        # temp = tempfile.NamedTemporaryFile()
        # for p in parses:
        #     temp.write(p)
        # temp.flush()
        # proc = subprocess.Popen(['java -mx100m -cp stanford-tregex/stanford-tregex.jar \
        #                           edu.stanford.nlp.trees.tregex.tsurgeon.Tsurgeon \
        #                           -s -treeFile %s surgery/*' % temp.name],
        #                         shell=True,
        #                         stdout=subprocess.PIPE,
        #                         stderr=subprocess.PIPE)
        # modparses = proc.communicate()[0].splitlines()
        # temp.close()
        modparses = modify_parses(parses)

        for i,chunk in enumerate(modparses[:]):
            for j,modparse in enumerate(chunk):
                if 'LANDMARK-PHRASE' in modparse:
                    modparses[i] = modparse
                    parses[i] = parses[i][j]
                    break
            if isinstance(modparses[i],list):
                modparses[i] = modparses[i][0]
                parses[i] = parses[i][0]


        for s,p,m in zip(all_descs,parses,modparses):
            print s,'\n',p,'\n',m,'\n\n'
            SentenceParse.add_sentence_parse(s,p,m)
        exit("Parsed everything")

    test_scenes = []

    for s in all_scenes:
        toremove = []
        test_scene = {'scene':s['scene'],
                      'speaker':s['speaker'],
                      'lmks':[],
                      'loc_descs':[],
                      'ids':[]}
        for i,(lmk,sentence_chunks, eyedee) in enumerate(zip(s['lmks'], s['loc_descs'], s['ids'])):
            original = list(sentence_chunks)
            # print i, lmk,
            # for l in sentence_chunks:
            #     print l,'--',
            # print
            for chunk in list(sentence_chunks):
                try:
                    parsetree, modparsetree = get_modparse(chunk)
                    # print modparsetree
                    # raw_input()
#                    if ('(NP' in modparsetree or '(PP' in modparsetree):
#                        sentence_chunks.remove(chunk)
                   # if 'objects' in chunk:
                   #     sentence_chunks.remove(chunk)
#                    elif (' side' in chunk or
#                          'end' in chunk or
#                          'edge' in chunk or
#                          'corner' in chunk or 
#                          'middle' in chunk or 
#                          'center' in chunk or
#                          'centre' in chunk) and not ('table' in chunk):
#                        sentence_chunks.remove(chunk)
#                    elif 'viewer' in chunk or 'between' in chunk:
#                        sentence_chunks.remove(chunk)
    
                except ParseError:
                    sentence_chunks.remove(chunk)
                    continue

            if eyedee in test_set:
                toremove.append(i)
                test_scene['lmks'].append(lmk)
                if len(sentence_chunks) == 0:
                    test_scene['loc_descs'].append(original)
                else:
                    test_scene['loc_descs'].append(sentence_chunks)
                test_scene['ids'].append(eyedee)
            elif len(sentence_chunks) == 0:
                toremove.append(i)
            
        test_scenes.append(test_scene)

        for i in reversed(toremove):
            try:
                no_bad_words.remove(s['ids'][i])
            except:
                pass
            del s['lmks'][i]
            del s['loc_descs'][i]
            del s['ids'][i]

    # all_meanings = speaker1.get_all_meanings(scene1)

    # s = all_scenes[0]
    # for lmk, loc_desc in zip(s['lmks'],s['loc_descs']):
    #     print lmk,
    #     for l in loc_desc:
    #         print l,'--',
    #     print

    # exit()

    # f = open('meaning_annotations.txt')
    # lines = f.readlines()
    # f.close()

    # scene0descs = []
    # for line in lines:
    #     line = line.strip()
    #     if line == '-'*48:
    #         # scenes.append(scene)
    #         # scene = []
    #         break
    #     else:
    #         user, annotation = line.split(' ---- ')
    #         user = [[x] for x in user.split(' -- ')]
    #         annotation = [[None if y.lower() == 'none' else y for y in x.split('; ')] for x in annotation.split(' -- ')]
    #         # print user,'----',annotation
    #         scene0descs.append((user,annotation))

    # data = all_scenes[0]

    # # if 'num_iterations' in data:
    # #     scene, speaker = construct_training_scene(True)
    # #     num_iterations = data['num_iterations']
    # # else:
    # scene = data['scene']
    # speaker = data['speaker']
    # num_iterations = len(data['loc_descs'])

    # # users,annotations = zip(*scene0descs)
    # # for obj, l, sentences, annotations in zip(data['lmks'], data['loc_descs'],users, annotations):
    # #     print obj, l, sentences, annotations
    # # exit()

    # step = 0.02
    # utils.scene.set_scene(scene,speaker)

    # scene_bb = scene.get_bounding_box()
    # scene_bb = scene_bb.inflate( Vec2(scene_bb.width*0.5,scene_bb.height*0.5) )
    # table = scene.landmarks['table'].representation.get_geometry()

    # # step = 0.04
    # loi = [lmk for lmk in scene.landmarks.values() if lmk.name != 'table']
    # all_heatmaps_tupless, xs, ys = speaker.generate_all_heatmaps(scene, step=step, loi=loi)

    # loi_infos = []
    # all_meanings = set()
    # for obj_lmk,all_heatmaps_tuples in zip(loi, all_heatmaps_tupless):

    #     lmks, rels, heatmapss = zip(*all_heatmaps_tuples)
    #     meanings = zip(lmks,rels)
    #     # print meanings
    #     all_meanings.update(meanings)
    #     loi_infos.append( (obj_lmk, meanings, heatmapss) )

    # all_heatmaps_tupless, xs, ys = speaker.generate_all_heatmaps(scene, step=step)
    # all_heatmaps_tuples = all_heatmaps_tupless[0]

    # object_meaning_applicabilities = {}
    # for obj_lmk, ms, heatmapss in loi_infos:
    #     for m,(h1,h2) in zip(ms, heatmapss):
    #         ps = [p for (x,y),p in zip(list(product(xs,ys)),h1) if obj_lmk.representation.contains_point( Vec2(x,y) )]
    #         if m not in object_meaning_applicabilities:
    #             object_meaning_applicabilities[m] = {}
    #         object_meaning_applicabilities[m][obj_lmk] = sum(ps)/len(ps)

    # # k = len(loi)
    # # for meaning_dict in object_meaning_applicabilities.values():
    # #     total = sum( meaning_dict.values() )
    # #     if total != 0:
    # #         for obj_lmk in meaning_dict.keys():
    # #             meaning_dict[obj_lmk] = meaning_dict[obj_lmk]/total - 1.0/k
    # #         total = sum( [value for value in meaning_dict.values() if value > 0] )
    # #         for obj_lmk in meaning_dict.keys():
    # #             meaning_dict[obj_lmk] = (2 if meaning_dict[obj_lmk] > 0 else 1)*meaning_dict[obj_lmk] - total

    # sorted_meaning_lists = {}

    # for m in object_meaning_applicabilities.keys():
    #     for obj_lmk in object_meaning_applicabilities[m].keys():
    #         if obj_lmk not in sorted_meaning_lists:
    #             sorted_meaning_lists[obj_lmk] = []
    #         sorted_meaning_lists[obj_lmk].append( (object_meaning_applicabilities[m][obj_lmk], m) )
    # for obj_lmk in sorted_meaning_lists.keys():
    #     sorted_meaning_lists[obj_lmk].sort(reverse=True)

    # users,annotations = zip(*scene0descs)
    # for trajector, sentences, annotations in zip(s['lmks'],users,annotations):
    #     probs, sorted_meanings = zip(*sorted_meaning_lists[trajector][:30])
    #     probs = np.array(probs)# - min(probs)
    #     probs /= probs.sum()

    #     for sentence,annotation in zip(sentences,annotations):
    #         print sentence, annotation
    #         for chunk in annotation:
    #             if chunk is not None:
    #                 try:
    #                     golden_posteriors = get_all_sentence_posteriors(chunk, all_meanings, golden=True, printing=False)
    #                 except ParseError as e:
    #                     logger( e )
    #                     prob = 0
    #                     rank = len(meanings)-1
    #                     entropy = 0
    #                     ed = len(sentence)
    #                     golden_log_probs.append( prob )
    #                     golden_entropies.append( entropy )
    #                     golden_ranks.append( rank )
    #                     min_dists.append( ed )
    #                     continue
    #                 epsilon = 1e-15
    #                 ps = [[golden_posteriors[lmk]*golden_posteriors[rel],(lmk,rel)] for lmk, rel in all_meanings]
    #                 ps = sorted(ps,reverse=True)
    #                 print chunk
    #                 for p,m in ps[:10]:
    #                     print p, m2s(*m)
    #                 raw_input()

        # for i,(p,sm) in enumerate(zip(probs[:15],sorted_meanings[:15])):
        #     lm,re = sm
        #     logger( '%i: %f %s' % (i,p,m2s(*sm)) )

    # # print 'good', good
    # print 'bad', bad

    total = 0
    print 'Train set:'
    for s in all_scenes:
        together = zip(s['loc_descs'],s['lmks'],s['ids'])
        # together = [(None,None,None)]*len(together)
        together = zip([None]*len(together),s['lmks'],s['ids'])
        shuffle(together)
        s['loc_descs'],s['lmks'],s['ids'] = zip(*together)
        print '  ',len(s['loc_descs']), len(s['lmks'])
        total+=len(s['lmks'])
    print '   total:',total
    total = 0
    print 'Test set:'
    for s in test_scenes:
        print '  ',len(s['loc_descs']), len(s['lmks'])
        total+=len(s['lmks'])
    print '   total:',total
    print len(no_bad_words)

    # f['all_scenes'] = all_scenes
    # f['test_scenes'] = test_scenes
    # f['test_set'] = test_set

    # all_scenes2 = f['all_scenes']
    # test_scenes2 = f['test_scenes']
    # test_set2 = f['test_set']
    # f.close()

    # for i,(scene_desc, test_scene_desc) in enumerate(zip(all_scenes2,all_scenes2)):

    #     scene = scene_desc['scene']
    #     speaker = scene_desc['speaker']
    #     assert(scene == test_scene_desc['scene'])
    #     assert(speaker == test_scene_desc['speaker'])

    testing_testing.autocorrect(
        all_scenes,
        test_scenes,
        test_set,
        scale=args.update_scale, 
        num_processors=args.num_processors, 
        num_samples=args.num_samples,
        tag=args.tag,
        step=0.02,
        chunksize=1)

    # object_correction_testing.autocorrect(1,
    #     scale=args.update_scale, 
    #     num_processors=args.num_processors, 
    #     num_samples=args.num_samples, 
    #     scene_descs=all_scenes,
    #     # learn_objects=True,
    #     # tag = args.tag,
    #     golden_metric=False, 
    #     mass_metric=False, 
    #     student_metric=False,
    #     choosing_metric=False, 
    #     step=0.04)

    # object_correction_testing.autocorrect(1,
    #     scale=args.update_scale, 
    #     num_processors=args.num_processors, 
    #     num_samples=args.num_samples, 
    #     scene_descs=test_scenes,
    #     learn_objects=False,
    #     tag = args.tag,
    #     golden_metric=False, 
    #     mass_metric=False, 
    #     student_metric=False,
    #     choosing_metric=False, 
    #     step=0.02)

#    print 'trained on:',len(all_scenes[0]['lmks'])+len(all_scenes[1]['lmks'])+len(all_scenes[2]['lmks'])+len(all_scenes[3]['lmks'])
#    print 'tested on:',len(all_scenes[4]['lmks'])
    print 'Trained on:',len(all_scenes[0]['lmks'])+len(all_scenes[1]['lmks'])+len(all_scenes[2]['lmks'])+len(all_scenes[3]['lmks'])+len(all_scenes[4]['lmks'])
    print 'Tested on:',len(test_scenes[0]['lmks'])+len(test_scenes[1]['lmks'])+len(test_scenes[2]['lmks'])+len(test_scenes[3]['lmks'])+len(test_scenes[4]['lmks'])
