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

from parse import parse_sentences, get_modparse
from models import SentenceParse

import object_correction_testing

import tempfile
import subprocess

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
        #print s.id, s.name
        for scene, speaker in run.read_scenes(os.path.join(args.scene_directory,s.name)):
            for t in tasks_descriptionquestion.query().filter(tasks_descriptionquestion.scene_id==s.id).all():
                #print t.id, t.scene_id, t.entity_id, t.answer, t.object_description, t.location_description
                entity_name = scenes_entity.query().filter(scenes_entity.id==t.entity_id).one().name
                lmk = scene.landmarks['object_%s' % entity_name]
                lmk_desc = t.object_description
                loc_desc = t.location_description
                # print lmk, entity_name, lmk_desc, loc_desc
                if loc_desc:
                    entity_names.append(entity_name)
                    lmks.append(lmk)
                    lmk_descs.append(lmk_desc)
                    loc_descs.append(loc_desc)
                    all_descs.append(loc_desc)

            all_scenes.append( [scene,speaker,lmks,loc_descs] )


    print 'loaded', len(all_scenes), 'scenes'

    sp_db = SentenceParse.get_sentence_parse(loc_descs[0])
    try:
        res = sp_db.all()[0]
    except IndexError:

        parses = parse_sentences(all_descs)

        temp = tempfile.NamedTemporaryFile()
        for p in parses:
            temp.write(p)
        temp.flush()
        proc = subprocess.Popen(['java -mx100m -cp stanford-tregex/stanford-tregex.jar \
                                  edu.stanford.nlp.trees.tregex.tsurgeon.Tsurgeon \
                                  -s -treeFile %s surgery/*' % temp.name],
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        modparses = proc.communicate()[0].splitlines()
        temp.close()

        for s,p,m in zip(all_descs,parses,modparses):
            SentenceParse.add_sentence_parse(s,p,m)

    for s in all_scenes:
        parses = []
        modparses = []
        for lmk,sentence in zip(s[2], s[3]):
            try:
                parse, modparse = get_modparse(sentence)
            except:
                s[2].remove(lmk)
                s[3].remove(sentence)

        print len(s[2])

    # scene, speaker = construct_training_scene()

    # object_correction_testing.autocorrect(args.num_iterations, # window=args.window_size,
    #     scale=args.update_scale, num_processors=args.num_processors, num_samples=args.num_samples, scene_descs=all_scenes)