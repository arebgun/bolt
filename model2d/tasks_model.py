#!/usr/bin/env python
# coding: utf-8

from __future__ import division

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.ext.declarative import _declarative_constructor
from sqlalchemy import func

from utils import force_unicode, bigrams, trigrams, lmk_id, logger

import numpy as np
from collections import defaultdict

import os
import sys
sys.path.append("..")

from semantics import run

from parse import parse_sentences, modify_parses, get_modparse, ParseError
from models import SentenceParse

import object_correction_testing

import tempfile
import subprocess
import re

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

    def __unicode__(self):
        return u'(%s,%s)' % (self.x, self.y)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-iterations', type=int, default=1)
    parser.add_argument('-u', '--update-scale', type=int, default=1000)
    parser.add_argument('-p', '--num-processors', type=int, default=7)
    parser.add_argument('-s', '--num-samples', action='store_true')
    parser.add_argument('-d', '--scene-directory', type=str)

    args = parser.parse_args()

    engine.echo = False
    create_all()

    # run.read_scenes(sys.argv[1])
    all_scenes   = []
    all_descs = []
    for s in scenes_scene.query().all():
        entity_names = []
        lmks         = []
        lmk_descs    = []
        loc_descs    = []
        ids = []
        #print s.id, s.name
        for scene, speaker in run.read_scenes(os.path.join(args.scene_directory,s.name), normalize=True):
            for t in tasks_descriptionquestion.query().filter(tasks_descriptionquestion.scene_id==s.id).all():
                #print t.id, t.scene_id, t.entity_id, t.answer, t.object_description, t.location_description
                entity_name = scenes_entity.query().filter(scenes_entity.id==t.entity_id).one().name
                lmk = scene.landmarks['object_%s' % entity_name]
                lmk_desc = t.object_description
                loc_desc = t.location_description
                eyedee = t.id
                # print lmk, entity_name, lmk_desc, loc_desc
                if loc_desc:

                    all_descs.append(loc_desc)

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
                                            # print '  ',parttt
                    # raw_input()

                    entity_names.append(entity_name)
                    lmks.append(lmk)
                    lmk_descs.append(lmk_desc)
                    # loc_descs.append(loc_desc)
                    # loc_descs.append(parttt.strip())
                    loc_descs.append(chunks)
                    ids.append( eyedee )

            all_scenes.append( {'scene':scene,'speaker':speaker,'lmks':lmks,'loc_descs':loc_descs, 'ids':ids} )


    print 'loaded', len(all_scenes), 'scenes'

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

    good = 0
    bad = 0
    for s in all_scenes:
        for lmk,sentence_chunks, eyedee in zip(s['lmks'], s['loc_descs'], s['ids']):

            for chunk in list(sentence_chunks):
                try:
                    parsetree, modparsetree = get_modparse(chunk)
                    # print modparsetree
                    # raw_input()
                    if ('(NP' in modparsetree or '(PP' in modparsetree):
                        sentence_chunks.remove(chunk)
                        pass
                    else:
                        if 'objects' in chunk:
                            bad += 1
                            # print ' '.join(sentence_chunks),'\n',chunk,'\n ', parsetree,'\n  ', modparsetree,'\n'
                            # raw_input()
                            sentence_chunks.remove(chunk)
                        elif (' side' in chunk or 
                           'corner' in chunk or 
                           'middle' in chunk or 
                           'center' in chunk or
                           'centre' in chunk):
                            if not ('table' in chunk):
                                sentence_chunks.remove(chunk)
                            else:
                                good += 1
                except ParseError:
                    sentence_chunks.remove(chunk)
                    continue

            if len(sentence_chunks) == 0:
                s['lmks'].remove(lmk)
                s['loc_descs'].remove(sentence_chunks)
                s['ids'].remove(eyedee)

    # # print 'good', good
    # print 'bad', bad

    # something = [3411,2701,1764,264,1142,852,2028,2774,3341,161,779,536,3853,357,249,2569,4175,2971,1368,3305,2586,591,3710,1909,
    # 4085,443,582,3895,3291,3535,2487,3204,476,3042,596,3524,1206,1284,4293,1168,410,3417,1332,892,623,2406,778,3472,2864,4174,1462,
    # 3416,453,4397,2036,1017,1033,3491,1207,4163,1092,2897,1445,2990,473,1155,510,1671,3957,561,1043,182,1854,238,2900,444,987,3041,
    # 612,1167,3594,2739,677,2585,2170,636,3940,2680,848,2938,2328,2829,3331,2871,2122,2169,3093,884,3557,525,3684,2277,4399,1135,1703,
    # 3268,3679,686,2448,992,3655,3711,1748,3881,3473,1756,3518,2376,3223,265,3026,3516,3749,1893,1496,4065,1280,378,3025,3348,820,1585,
    # 2257,3146,3147,3094,3198,3064,4189,693,1707,3831,3606,2758,988,4126,4352,2218,883,3288,2620,2227,2745,2775,4405,1061,2907,2753,3027,
    # 370,3904,2401,3555,978,1958,1930,3038,2439,962,491,4317,2314,2334,2638,3312,3724,3455,3435,543,2444,3997,4049,3935,3842,1527,735,4053,
    # 4135,3539,730,2482,3188,825,3037,4395,3657,428,3205,1490,3805]


    # thing = set()
    for s in all_scenes:
    #     for loc_desc in s['loc_descs']:
    #         for l in loc_desc:
    #             thing.add(l)

        # s['loc_descs'] = [None]*10#len(s['loc_descs'])
        # for eyedee in s['ids']:
        #     if eyedee in something:
        #         print eyedee
        # for loc_desc in s['loc_descs']:           
    #         print '--'.join(loc_desc)
        print len(s['loc_descs']), len(s['lmks'])
        # s['loc_descs'] = s['loc_descs'][:10]
        # s['lmks'] = s['lmks'][:10]
        # s['ids'] = s['ids'][:10]
    # exit()
    # how_many = 100
    # for s in all_scenes:
    #     s['lmks'] = s['lmks'][:how_many]
    #     s['loc_descs'] = s['loc_descs'][:how_many]
    #     s['ids'] = s['ids'][:how_many]


    # scene, speaker = construct_training_scene()

    object_correction_testing.autocorrect(1,
        scale=args.update_scale, 
        num_processors=args.num_processors, 
        num_samples=args.num_samples, 
        scene_descs=all_scenes,
        golden_metric=False, 
        mass_metric=False, 
        student_metric=False,
        choosing_metric=False, 
        step=0.02)