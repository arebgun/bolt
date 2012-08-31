#!/usr/bin/env python
# coding: utf-8

from __future__ import division

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.ext.declarative import _declarative_constructor

from utils import force_unicode, bigrams, trigrams, lmk_id

import numpy as np


### configuration ###

db_url = 'sqlite:///table2d.db'
#db_url = 'postgresql+pg8000://postgres:postgres@localhost:5432/table2d'
echo = False



### utilities ###

# as seen on http://stackoverflow.com/a/1383402
class ClassProperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

def create_all():
    """create all tables"""
    Base.metadata.create_all(engine)






### setting up sqlalchemy stuff ###

engine = create_engine(db_url, echo=echo)
Session = sessionmaker()
Session.configure(bind=engine)
session = Session()

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
    @ClassProperty
    @classmethod
    def query(cls):
        return session.query(cls)

    # like in elixir
    @classmethod
    def get_by(cls, **kwargs):
        return cls.query.filter_by(**kwargs).first()

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





### models start here ###

class Location(Base):
    id = Column(Integer, primary_key=True)

    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)

    # has many
    words = relationship('Word', backref='location')
    productions = relationship('Production', backref='location')

    def __unicode__(self):
        return u'(%s,%s)' % (self.x, self.y)


class Word(Base):
    id = Column(Integer, primary_key=True)

    word = Column(String, nullable=False)
    pos = Column(String, nullable=False)

    # belongs to
    parent_id = Column(Integer, ForeignKey('productions.id'))
    location_id = Column(Integer, ForeignKey('locations.id'))

    def __unicode__(self):
        return force_unicode(self.word)

    @classmethod
    def get_words(cls, pos=None, lmk=None, rel=None):
        q = cls.query.join(Production)

        if pos is not None:
            q = q.filter(Word.pos==pos)

        if lmk is not None:
            q = q.filter(Production.landmark==lmk)

        if rel is not None:
            q = q.filter(Production.relation==rel)

        return q

    @classmethod
    def delete_words(cls, limit, pos, word, lmk=None, rel=None):
        q = cls.query.join(Production)

        if pos is not None:
            q = q.filter(Word.pos==pos)

        if lmk is not None:
            q = q.filter(Production.landmark==lmk)

        if rel is not None:
            q = q.filter(Production.relation==rel)

        q.filter(Word.word==word)

        return q.limit(limit).delete()

class CWord(Base):
    id = Column(Integer, primary_key=True)

    word = Column(String, nullable=False)
    pos = Column(String, nullable=False)

    landmark = Column(Integer)
    landmark_class = Column(String)
    landmark_orientation_relations = Column(String)
    landmark_color = Column(String)
    relation = Column(String)
    relation_distance_class = Column(String)
    relation_degree_class = Column(String)

    count = Column(Float, nullable=False, default=0)

    @classmethod
    def get_word_counts(cls,
                        pos=None,
                        word=None,
                        lmk=None,
                        lmk_class=None,
                        lmk_ori_rels=None,
                        lmk_color=None,
                        rel=None,
                        rel_dist_class=None,
                        rel_deg_class=None):
        q = cls.query
        if word != None:
            q = q.filter(CWord.word==word)
        if pos != None:
            q = q.filter(CWord.pos==pos)
        if lmk != None:
            q = q.filter(CWord.landmark==lmk)
        if lmk_class != None:
            q = q.filter(CWord.landmark_class==lmk_class)
        if lmk_ori_rels is not None:
            q = q.filter(CWord.landmark_orientation_relations==lmk_ori_rels)
        if lmk_color is not None:
            q = q.filter(CWord.landmark_color==lmk_color)
        if rel != None:
            q = q.filter(CWord.relation==rel)
        if rel_dist_class != None:
            q = q.filter(CWord.relation_distance_class==rel_dist_class)
        if rel_deg_class != None:
            q = q.filter(CWord.relation_degree_class==rel_deg_class)

        return q

    @classmethod
    def update_word_counts(cls,
                           update,
                           pos,
                           word,
                           lmk=None,
                           lmk_class=None,
                           lmk_ori_rels=None,
                           lmk_color=None,
                           rel=None,
                           rel_dist_class=None,
                           rel_deg_class=None):
        cp_db = cls.get_word_counts(pos, word, lmk, lmk_class, lmk_ori_rels, lmk_color, rel, rel_dist_class, rel_deg_class)

        if cp_db.count() <= 0:
            assert(update > 0)
            CWord(word=word,
                  pos=pos,
                  landmark=lmk_id(lmk),
                  landmark_class=lmk_class,
                  landmark_orientation_relations=lmk_ori_rels,
                  landmark_color=lmk_color,
                  relation=rel,
                  relation_distance_class=rel_dist_class,
                  relation_degree_class=rel_deg_class,
                  count=update)
        else:
            # for cword in cp_db.all():
            #     print 'Count for %s before: %f' % (cword.word, cword.count)
            #     cword.count *= (1.0 + update)
            #     print 'Count for %s after: %f' % (cword.word, cword.count)

            ccounter = {}
            for cword in cp_db.all():
                # print cword.word, cword.count
                if cword.word in ccounter: ccounter[cword.word] += cword.count
                else: ccounter[cword.word] = cword.count

            # print '----------------'

            ckeys, ccounts = zip(*ccounter.items())

            ccounts = np.array(ccounts, dtype=float)
            ccounts /= ccounts.sum()
            updates = ccounts * update
            ups = dict( zip(ckeys, updates) )

            for cword in cp_db.all():
                if cword.count <= -ups[cword.word]: cword.count = 1
                else: cword.count += ups[cword.word]
                # print cword.word, cword.count

        session.commit()

    def __unicode__(self):
        return u'%s (%s)' % (self.word, self.count)

class Bigram(Base):
    id = Column(Integer, primary_key=True)

    w1_id = Column(Integer, ForeignKey('words.id'))
    w2_id = Column(Integer, ForeignKey('words.id'))

    w1 = relationship('Word', primaryjoin='Word.id==Bigram.w1_id')
    w2 = relationship('Word', primaryjoin='Word.id==Bigram.w2_id')

    def __unicode__(self):
        return u'%s %s' % (self.w1, self.w2)

    @classmethod
    def make_bigrams(cls, words):
        for w1,w2 in bigrams(words):
            bigram = cls()
            if isinstance(w1, Word):
                bigram.w1 = w1
            if isinstance(w2, Word):
                bigram.w2 = w2

class Trigram(Base):
    id = Column(Integer, primary_key=True)

    w1_id = Column(Integer, ForeignKey('words.id'))
    w2_id = Column(Integer, ForeignKey('words.id'))
    w3_id = Column(Integer, ForeignKey('words.id'))

    w1 = relationship('Word', primaryjoin='Word.id==Trigram.w1_id')
    w2 = relationship('Word', primaryjoin='Word.id==Trigram.w2_id')
    w3 = relationship('Word', primaryjoin='Word.id==Trigram.w3_id')

    def __unicode__(self):
        return u'%s %s %s' % (self.w1, self.w2, self.w3)

    @classmethod
    def make_trigrams(cls, words):
        for w1,w2,w3 in trigrams(words):
            trigram = cls()
            if isinstance(w1, Word):
                trigram.w1 = w1
            if isinstance(w2, Word):
                trigram.w2 = w2
            if isinstance(w3, Word):
                trigram.w3 = w3

class Production(Base):
    id = Column(Integer, primary_key=True)

    lhs = Column(String, nullable=False)
    rhs = Column(String, nullable=False)

    # semantic content
    landmark = Column(Integer)
    landmark_class = Column(String)
    landmark_orientation_relations = Column(String)
    landmark_color = Column(String)
    relation = Column(String)
    relation_distance_class = Column(String)
    relation_degree_class = Column(String)

    # belongs_to
    parent_id = Column(Integer, ForeignKey('productions.id'))
    location_id = Column(Integer, ForeignKey('locations.id'))

    # has many
    words = relationship('Word', backref='parent')
    productions = relationship('Production', backref=backref('parent',
                                                             remote_side=[id]))

    def __unicode__(self):
        return u'%s -> %s' % (self.lhs, self.rhs)

    @classmethod
    def get_productions(cls, lhs=None, parent=None, lmk=None, rel=None):
        q = cls.query

        if lhs is not None:
            q = q.filter(Production.lhs==lhs)

        if lmk is not None:
            q = q.filter(Production.landmark==lmk)

        if rel is not None:
            q = q.filter(Production.relation==rel)

        if parent is not None:
            q = q.join(Production.parent, aliased=True).\
                  filter(Production.lhs==parent).\
                  reset_joinpoint()

        return q


class CProduction(Base):
    id = Column(Integer, primary_key=True)

    lhs = Column(String, nullable=False)
    rhs = Column(String, nullable=False)
    parent = Column(String)

    # semantic content
    landmark = Column(Integer)
    landmark_class = Column(String)
    landmark_orientation_relations = Column(String)
    landmark_color = Column(String)
    relation = Column(String)
    relation_distance_class = Column(String)
    relation_degree_class = Column(String)

    count = Column(Float, nullable=False, default=0)

    def __unicode__(self):
        return u'%s -> %s (%s)' % (self.lhs, self.rhs, self.count)

    @classmethod
    def get_production_counts(cls,
                              lhs,
                              rhs=None,
                              parent=None,
                              lmk_class=None,
                              lmk_ori_rels=None,
                              lmk_color=None,
                              rel=None,
                              dist_class=None,
                              deg_class=None):
        q = cls.query
        if lhs != None:
            q = q.filter(CProduction.lhs==lhs)
        if rhs != None:
            q = q.filter(CProduction.rhs==rhs)
        if parent != None:
            q = q.filter(CProduction.parent==parent)
        if lmk_class != None:
            q = q.filter(CProduction.landmark_class==lmk_class)
        if lmk_ori_rels is not None:
            q = q.filter(CProduction.landmark_orientation_relations==lmk_ori_rels)
        if lmk_color is not None:
            q = q.filter(CProduction.landmark_color==lmk_color)
        if rel != None:
            q = q.filter(CProduction.relation==rel)
        if dist_class != None:
            q = q.filter(CProduction.relation_distance_class==dist_class)
        if deg_class != None:
            q = q.filter(CProduction.relation_degree_class==deg_class)

        return q

    @classmethod
    def update_production_counts(cls,
                                 update,
                                 lhs,
                                 rhs,
                                 parent=None,
                                 lmk_class=None,
                                 lmk_ori_rels=None,
                                 lmk_color=None,
                                 rel=None,
                                 dist_class=None,
                                 deg_class=None):
        cp_db = cls.get_production_counts(lhs,rhs,parent,lmk_class,lmk_ori_rels,lmk_color,rel,dist_class,deg_class)

        if cp_db.count() <= 0:
            assert(update > 0)
            CProduction(lhs=lhs,
                        rhs=rhs,
                        parent=parent,
                        landmark_class=lmk_class,
                        landmark_orientation_relations=lmk_ori_rels,
                        landmark_color=lmk_color,
                        relation=rel,
                        relation_distance_class=dist_class,
                        relation_degree_class=deg_class,
                        count=update)
        else:
            # for cprod in cp_db.all():
            #     print 'Count for %s before: %f' % (cprod.rhs, cprod.count)
            #     cprod.count *= (1.0 + update)
            #     print 'Count for %s after: %f' % (cprod.rhs, cprod.count)

            ccounter = {}
            for cprod in cp_db.all():
                # print cprod.rhs, cprod.count
                if cprod.rhs in ccounter: ccounter[cprod.rhs] += cprod.count
                else: ccounter[cprod.rhs] = cprod.count

            # print '----------------'

            ckeys, ccounts = zip(*ccounter.items())

            ccounts = np.array(ccounts, dtype=float)
            ccounts /= ccounts.sum()
            updates = ccounts * update
            ups = dict( zip(ckeys, updates) )

            for cprod in cp_db.all():
                if cprod.count <= -ups[cprod.rhs]: cprod.count = 1
                else: cprod.count += ups[cprod.rhs]
                # print cprod.rhs, cprod.count

        session.commit()

class WordCPT(Base):
    id = Column(Integer, primary_key=True)

    word = Column(String, nullable=False)
    all_count = Column(Float)
    count = Column(Float)
    prob = Column(Float)

    # conditioned on
    pos = Column(String)
    lmk = Column(Integer)
    rel = Column(String)

    fields = ['pos', 'lmk', 'rel']

    def __unicode__(self):
        given = [(f,getattr(self,f)) for f in self.fields if getattr(self,f) is not None]
        given = ', '.join(u'%s=%r' % g for g in given)
        if given:
            return u'Pr(word=%r | %s) = %s' % (self.word, given, self.prob)
        else:
            return u'Pr(word=%r) = %s' % (self.word, self.prob)

    @classmethod
    def calc_prob(cls, word, **given):
        """calculates conditional probability"""
        wp = WordCPT(word=word, **given)
        q = Word.get_words(**given)
        const = q.count()  # normalizing constant
        if const:
            wp.all_count = const
            wp.count = q.filter(Word.word==word).count()
            wp.prob = wp.count / const
        return wp

    @classmethod
    def get_prob(cls, word, **given):
        """gets probability from db"""
        params = dict((f,None) for f in cls.fields)
        params.update(given)
        return cls.query.filter_by(word=word, **given).one()

    @classmethod
    def probability(cls, word, **given):
        try:
            wp = cls.get_prob(word=word, **given)
        except:
            wp = cls.calc_prob(word=word, **given)
            session.commit()
        return wp.count / wp.all_count

    @classmethod
    def update(cls, word, update_by, **given):
        try:
            wp = cls.get_prob(word=word, **given)
        except:
            wp = cls.calc_prob(word=word, **given)
        wp.all_count = wp.all_count + update_by
        wp.count = wp.count + update_by
        session.commit()
        return

class ExpansionCPT(Base):
    id = Column(Integer, primary_key=True)

    rhs = Column(String, nullable=False)
    all_count = Column(Float)
    count = Column(Float)
    prob = Column(Float)

    # conditioned on
    lhs = Column(String)
    parent = Column(String)
    lmk = Column(Integer)
    rel = Column(String)

    fields = ['lhs', 'parent', 'lmk', 'rel']

    def __unicode__(self):
        given = [(f,getattr(self,f)) for f in self.fields if getattr(self,f) is not None]
        given = ', '.join(u'%s=%r' % g for g in given)
        if given:
            return u'Pr(rhs=%r | %s) = %s' % (self.rhs, given, self.prob)
        else:
            return u'Pr(rhs=%r) = %s' % (self.rhs, self.prob)

    @classmethod
    def calc_prob(cls, rhs, **given):
        """calculates conditional probability"""
        ep = ExpansionCPT(rhs=rhs, **given)
        q = Production.get_productions(**given)
        const = q.count()  # normalizing constant
        if const:
            ep.all_count = const
            ep.count = q.filter_by(rhs=rhs).count()
            ep.prob = ep.count / const
        return ep

    @classmethod
    def get_prob(cls, rhs, **given):
        """gets probability stored in db"""
        params = dict((f, None) for f in cls.fields)
        params.update(given)
        return cls.query.filter_by(rhs=rhs, **params).one()

    @classmethod
    def probability(cls, rhs, **given):
        try:
            ep = cls.get_prob(rhs=rhs, **given)
        except:
            ep = cls.calc_prob(rhs=rhs, **given)
            session.commit()
        return ep.count / ep.all_count

    @classmethod
    def update(cls, rhs, update_by, **given):
        try:
            ep = cls.get_prob(rhs=rhs, **given)
        except:
            ep = cls.calc_prob(rhs=rhs, **given)
        ep.all_count = ep.all_count + update_by
        ep.count = ep.count + update_by
        session.commit()
        return


class SentenceParse(Base):
    id = Column(Integer, primary_key=True)

    sentence = Column(String)
    original_parse = Column(String)
    modified_parse = Column(String)

    @classmethod
    def get_sentence_parse(cls, sentence, orig_parse=None, mod_parse=None):
        q = cls.query
        q = q.filter(SentenceParse.sentence==sentence)

        # if orig_parse != None:
        #     q = q.filter(SentenceParses.original_parse==orig_parse)

        return q

    @classmethod
    def add_sentence_parse(cls, sentence, orig_parse, mod_parse):
        sp_db = cls.get_sentence_parse(sentence, orig_parse, mod_parse)
        if sp_db.count() <= 0:
            SentenceParse(sentence=sentence,
                          original_parse=orig_parse,
                          modified_parse=mod_parse)
            session.commit()

    @classmethod
    def add_sentence_parse_blind(cls, sentence, orig_parse, mod_parse):
        SentenceParse(sentence=sentence,
                      original_parse=orig_parse,
                      modified_parse=mod_parse)



if __name__ == '__main__':
    engine.echo = True
    create_all()
