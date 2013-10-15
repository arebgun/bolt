#!/usr/bin/env python
# coding: utf-8

from __future__ import division

from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship, backref, sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.ext.declarative import _declarative_constructor
from sqlalchemy import func

from utils import force_unicode, bigrams, trigrams, lmk_id, logger

import numpy as np
import math


### configuration ###

db_url = 'sqlite:///table2d.db'
golden_db_url = 'sqlite:///golden-table2d.db'
# db_url = 'postgresql+psycopg2://postgres:password@localhost:5432/table2d'
# golden_db_url = 'postgresql+psycopg2://postgres:password@localhost:5432/golden-table2d'
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
# Session = sessionmaker()
session = scoped_session(sessionmaker())
session.configure(bind=engine)
# session = Session()

golden_engine = create_engine(golden_db_url, echo=echo)
# golden_Session = sessionmaker()
golden_session = scoped_session(sessionmaker())
golden_session.configure(bind=golden_engine)
# golden_session = golden_Session()

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
        if 'golden' in kwargs and kwargs['golden']:
            return golden_session().query(cls, *args)
        else:
            return session().query(cls, *args)

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
        session().add(self)

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
        q = cls.query().join(Production)

        if pos is not None:
            q = q.filter(Word.pos==pos)

        if lmk is not None:
            q = q.filter(Production.landmark==lmk)

        if rel is not None:
            q = q.filter(Production.relation==rel)

        return q

    @classmethod
    def delete_words(cls, limit, pos, word, lmk=None, rel=None):
        q = cls.query().join(Production)

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
    prev_word = Column(String)

    count = Column(Float, nullable=False, default=0)

    @classmethod
    def get_count(cls, **kwargs):
        q = cls.get_word_counts(**kwargs)
        return sum(c.count for c in q)

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
                        rel_deg_class=None,
                        prev_word='<no prev word>',
                        golden=False):
        q = cls.query(golden=golden)
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
        # NOTE `None` is a valid value for `prev_word`, it means the current
        # word is the beginning of the sentence.
        if prev_word != '<no prev word>':
            q = q.filter(CWord.prev_word==prev_word)

        return q

    @classmethod
    def get_word_sum(cls,
                        pos=None,
                        word=None,
                        lmk=None,
                        lmk_class=None,
                        lmk_ori_rels=None,
                        lmk_color=None,
                        rel=None,
                        rel_dist_class=None,
                        rel_deg_class=None,
                        prev_word='<no prev word>',
                        golden=False):
        if golden:
            q = golden_session().query(func.sum(CWord.count))
        else:
            q = session().query(func.sum(CWord.count))

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
        # NOTE `None` is a valid value for `prev_word`, it means the current
        # word is the beginning of the sentence.
        if prev_word != '<no prev word>':
            q = q.filter(CWord.prev_word==prev_word)

        return q.scalar()

    @classmethod
    def update_word_counts(cls,
                           update,
                           pos,
                           word,
                           prev_word,
                           lmk=None,
                           lmk_class=None,
                           lmk_ori_rels=None,
                           lmk_color=None,
                           rel=None,
                           rel_dist_class=None,
                           rel_deg_class=None,
                           golden=False,
                           multiply=False):

        # logger( 'Really gonna multiply??? %s' % multiply, 'okgreen' )
        # if multiply:
        #     cp_db = cls.get_word_counts(pos=pos,
        #                                 lmk=lmk,
        #                                 lmk_class=lmk_class,
        #                                 lmk_ori_rels=lmk_ori_rels,
        #                                 lmk_color=lmk_color,
        #                                 rel=rel,
        #                                 rel_dist_class=rel_dist_class,
        #                                 rel_deg_class=rel_deg_class,
        #                                 prev_word=prev_word,
        #                                 golden=golden)
        #     if cp_db.count() <= 0:
        #         update *= 10
        #         # logger( 'Count was zero', 'okgreen' )
        #     else:
        #         ccounter = defaultdict(int)
        #         ccounter[word] = 0
        #         for cword in cp_db.all():
        #             ccounter[cword.word] += cword.count

        #         ckeys, ccounts = zip(*ccounter.items())
        #         ccounts = np.array(ccounts, dtype=float)
        #         total = ccounts.sum()
        #         update *= total

        cp_db = cls.get_word_counts(pos=pos,
                                    word=word,
                                    lmk=lmk,
                                    lmk_class=lmk_class,
                                    lmk_ori_rels=lmk_ori_rels,
                                    lmk_color=lmk_color,
                                    rel=rel,
                                    rel_dist_class=rel_dist_class,
                                    rel_deg_class=rel_deg_class,
                                    prev_word=prev_word,
                                    golden=golden)

        committed = False
        while not committed:

            try:
                num_results = cp_db.count()
                if num_results <= 0:
                    if update <= 0: return
                    # logger( 'Updating by %f, %f' % (update, update), 'warning')
                    count = update
                    CWord(word=word,
                          pos=pos,
                          prev_word=prev_word,
                          landmark=lmk_id(lmk),
                          landmark_class=lmk_class,
                          landmark_orientation_relations=lmk_ori_rels,
                          landmark_color=lmk_color,
                          relation=rel,
                          relation_distance_class=rel_dist_class,
                          relation_degree_class=rel_deg_class,
                          count=count)

                # elif num_results == 1:

                #     cword = cp_db.one()
                #     if multiply:
                #         # logger( 'Updating by %f, %f' % (update, ups[cword.word]), 'warning')
                #         cword.count *= 1+update
                #         if cword.count < 1: cword.count = 1
                #     else:
                #         # logger( 'Updating by %f, %f' % (update, ups[cword.word]), 'warning')
                #         if cword.count <= -update: cword.count = 1
                #         else: cword.count += update

                else:

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

                    if multiply:
                        for cword in cp_db.all():
                            # logger( 'Updating by %f, %f' % (update, ups[cword.word]), 'warning')
                            assert( not np.isnan( ups[cword.word] ) )
                            cword.count *= 1+ups[cword.word]
                            if cword.count < 1: cword.count = 1
                    else:
                        for cword in cp_db.all():
                            # logger( 'Updating by %f, %f' % (update, ups[cword.word]), 'warning')
                            if cword.count <= -ups[cword.word]: cword.count = 1
                            else: cword.count += ups[cword.word]

                session().commit()
                committed = True
            except Exception as e:
                logger( 'Could not commit', 'warning' )
                logger( e )
                session().rollback()
                continue

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
    parent_id = Column(String)
    location_id = Column(Integer, ForeignKey('locations.id'))

    # has many
    words = relationship('Word', backref='parent')
    # productions = relationship('Production', backref=backref('parent',
    #                                                          remote_side=[id]))

    def __unicode__(self):
        return u'%s -> %s' % (self.lhs, self.rhs)

    @classmethod
    def get_productions(cls, lhs=None, parent=None, lmk=None, rel=None):
        q = cls.query()

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
        # return u'%s -> %s (%s)' % (self.lhs, self.rhs, self.count)
        return u'[%d] %s -> %s p=[%s] lmk=[%s %s %s] rel=[%s %s %s]' % (self.count, self.lhs, self.rhs, self.parent, self.landmark_class, self.landmark_color, self.landmark_orientation_relations, self.relation, self.relation_degree_class, self.relation_distance_class)

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
                              deg_class=None,
                              golden=False):
        q = cls.query(golden=golden)
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
    def get_productions_context(cls,
                                lhs=None,
                                rhs=None,
                                parent=None,
                                relation=None,
                                relation_distance_class=None,
                                relation_degree_class=None,
                                landmark_class=None,
                                landmark_color=None,
                                landmark_orientation_relations=None,
                                golden=False):
        q = cls.query(CProduction.lhs,
                      CProduction.rhs,
                      CProduction.parent,
                      CProduction.relation,
                      CProduction.relation_distance_class,
                      CProduction.relation_degree_class,
                      CProduction.landmark_class,
                      CProduction.landmark_color,
                      CProduction.landmark_orientation_relations,
                      func.sum(CProduction.count).label('count'),
                      golden=golden)

        if lhs is not None: q = q.filter(CProduction.lhs==lhs)
        if rhs is not None: q = q.filter(CProduction.rhs==rhs)
        if parent is not None: q = q.filter(CProduction.parent==parent)
        if relation is not None: q = q.filter(CProduction.relation==relation)
        if relation_distance_class is not None: q = q.filter(CProduction.relation_distance_class==relation_distance_class)
        if relation_degree_class is not None: q = q.filter(CProduction.relation_degree_class==relation_degree_class)
        if landmark_class is not None: q = q.filter(CProduction.landmark_class==landmark_class)
        if landmark_color is not None: q = q.filter(CProduction.landmark_color==landmark_color)
        if landmark_orientation_relations is not None: q = q.filter(CProduction.landmark_orientation_relations==landmark_orientation_relations)
        q = q.group_by(CProduction.rhs, CProduction.parent)
        q = q.order_by('count DESC')

        return q.all()

    @classmethod
    def get_production_sum(cls,
                          lhs,
                          rhs=None,
                          parent=None,
                          lmk_class=None,
                          lmk_ori_rels=None,
                          lmk_color=None,
                          rel=None,
                          dist_class=None,
                          deg_class=None,
                          golden=False):
        if golden:
            q = golden_session().query(func.sum(CProduction.count))
        else:
            q = session().query(func.sum(CProduction.count))
        q = q.filter(CProduction.lhs!='LOCATION-PHRASE')

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

        return q.scalar()

    @classmethod
    def get_entropy_ratio(cls, lhs, rhs, column, parent=None, golden=False, verbose=False):
        if verbose: print 'Calculating entropy ratio for %s column' % column

        # column entropy unconditinal
        z = cls.query(column,
                      func.sum(CProduction.count).label('count'),
                      golden=golden) \
                .filter(column!=None) \
                .filter(column!=',,,') \
                .group_by(column) \
                .all()

        max_entropy = 0
        max_sum = sum([prod.count for prod in z])
        if verbose: print 'total count (unconditional):', max_sum

        for prod in z:
            p = prod.count / max_sum
            max_entropy -= p * math.log(p)
            if verbose: print prod

        if verbose:
            print 'column entropy (unconditional):', max_entropy
            print '*'*70

        # column entropy conditioned on lhs and rhs
        q = cls.query(CProduction.lhs,
                      CProduction.rhs,
                      CProduction.parent,
                      column,
                      func.sum(CProduction.count).label('count'),
                      golden=golden) \
                .filter(CProduction.lhs==lhs) \
                .filter(column!=None) \
                .filter(column!=',,,')

        if rhs is not None: q = q.filter(CProduction.rhs==rhs)
        if parent is not None: q = q.filter(CProduction.parent==parent)

        q = q.group_by(column, CProduction.parent)

        q.all()

        cond_entropy = 0
        cond_sum = sum([res.count for res in q])
        if verbose: print 'total count (conditioned on lhs: %s, rhs: %s and parent: %s):' % (lhs,rhs,parent), cond_sum

        for prod in q:
            p = prod.count / cond_sum
            cond_entropy -= p * math.log(p)
            if verbose: print prod

        ratio = cond_entropy / max_entropy
        if cond_sum == 0: ratio = max_entropy

        if verbose:
            print 'column entropy (conditioned on lhs: %s, rhs: %s and parent: %s):' % (lhs,rhs,parent), cond_entropy
            print 'RATIO:', ratio

        return ratio


    @classmethod
    def get_entropy_ratio_sample_dependent(cls, lhs, rhs, column, parent=None, golden=False, verbose=False):

        def laplace_estimator(f, N, n):
            return (N * f + 1) / float( N + n)

        def entropy(N, ps, worst_ps):
            nom = [laplace_estimator( f, N, len(ps) ) * ( 1 - laplace_estimator( f, N, len(ps) ) )  for f in ps]
            denom = [ (-2 * w + 1) * laplace_estimator(w, N, len(worst_ps) ) + w * 2 for w in worst_ps]
            return sum([ni / denomi for ni,denomi in zip(nom, denom)])

        if verbose: print 'Calculating entropy ratio for %s column' % column

        # column entropy unconditinal
        z = cls.query(column,
                      func.sum(CProduction.count).label('count'),
                      golden=golden) \
                .filter(column!=None) \
                .filter(column!=',,,') \
                .group_by(column) \
                .all()

        attr_idx = str(column).find('.')+1
        attr_name = str(column)[attr_idx:]
        z_cols = [getattr(row, attr_name) for row in z]
        z_counts = [row.count for row in z]

        # column entropy conditioned on lhs and rhs
        q = cls.query(CProduction.lhs,
                      CProduction.rhs,
                      CProduction.parent,
                      column,
                      func.sum(CProduction.count).label('count'),
                      golden=golden) \
                .filter(CProduction.lhs==lhs) \
                .filter(column!=None) \
                .filter(column!=',,,')

        if rhs is not None: q = q.filter(CProduction.rhs==rhs)
        if parent is not None: q = q.filter(CProduction.parent==parent)

        q = q.group_by(column, CProduction.parent)
        q.all()

        q_cols = [getattr(row, attr_name) for row in q]
        q_counts = [row.count for row in q]

        q_c = [1e-6] * len(z_counts)

        for col,count in zip(q_cols, q_counts):
            try:
                index = z_cols.index(col)
                q_c[index] = count
            except:
                pass

        return entropy(sum(q_c), [c/float(sum(q_c)) for c in q_c], [c/float(sum(z_counts)) for c in z_counts])


    @classmethod
    def get_mutual_information(cls, lhs, rhs, column, golden=False, verbose=False):
        if verbose: print 'Calculating mutual information for %s column' % column

        # unconditinal
        z = cls.query(column,
                      func.sum(CProduction.count).label('count'),
                      golden=golden) \
                .filter(column!=None) \
                .filter(column!=',,,') \
                .group_by(column) \
                .all()

        max_sum = sum([prod.count for prod in z])

        uncond = []
        for prod in z:
            p = prod.count / float(max_sum)
            uncond.append( (prod, p) )

        # conditioned on lhs and rhs
        q = cls.query(CProduction.lhs,
                      CProduction.rhs,
                      CProduction.parent,
                      column,
                      func.sum(CProduction.count).label('count'),
                      golden=golden) \
                .filter(CProduction.lhs==lhs) \
                .filter(column!=None) \
                .filter(column!=',,,')

        if rhs is not None: q = q.filter(CProduction.rhs==rhs)
        q = q.group_by(column, CProduction.parent)
        q.all()

        max_sum = sum([prod.count for prod in q])

        cond = []
        for prod in q:
            p = prod.count / float(max_sum)
            cond.append( (prod, p) )

        # joint
        joint_ditribution = []
        attr_idx = str(column).find('.')+1
        attr_name = str(column)[attr_idx:]

        for x_row,_ in cond:
            # print x_row.lhs, x_row.rhs, x_row.parent, getattr(x_row, attr_name), x_row.count
            for y_row,_ in uncond:
                # print '\t', getattr(y_row, attr_name), y_row.count
                qq = cls.query(CProduction.lhs,
                      CProduction.rhs,
                      CProduction.parent,
                      column,
                      func.sum(CProduction.count).label('count'),
                      golden=golden)

                if x_row.lhs is not None: qq = qq.filter(CProduction.lhs==x_row.lhs)
                if x_row.rhs is not None: qq = qq.filter(CProduction.rhs==x_row.rhs)
                if x_row.parent is not None: qq = qq.filter(CProduction.parent==x_row.parent)

                attr_value = getattr(y_row, attr_name)
                if attr_value is not None: qq = qq.filter(column==attr_value)

                v = qq.one()
                if v.count is not None:
                    joint_ditribution.append( (x_row, y_row, v) )

                    # print '\t\t%s -> %s [%s], %s -- %d' % (v.lhs, v.rhs, v.parent, getattr(v, attr_name), v.count)

        max_sum = sum([prod.count for _,_,prod in joint_ditribution])

        joint = {}
        for x,y,prod in joint_ditribution:
            p = prod.count / float(max_sum)
            joint[(x,y)] = (prod, p)

        # x = lhs,rhs,parent
        # y = column
        # x,y = lhs,rhs,parent,column
        mi = 0
        for x,p_x in cond:
            for y,p_y in uncond:
                if (x,y) in joint:
                    _,j = joint[(x,y)]
                    mi += j*math.log(j/(p_x*p_y))

        return 1.0/mi if mi > 0 else 99

    @classmethod
    def get_probability(cls, lhs, rhs, column, column_value, parent=None, golden=False, verbose=False):
        q = cls.query(CProduction.lhs,
                      CProduction.rhs,
                      CProduction.parent,
                      column,
                      func.sum(CProduction.count).label('count'),
                      golden=golden) \
                .filter(CProduction.lhs==lhs) \
                .filter(column==column_value)#!=None) \
                # .filter(column!=',,,')

        if rhs is not None: q = q.filter(CProduction.rhs==rhs)
        if parent is not None: q = q.filter(CProduction.parent==parent)

        q = q.group_by(column, CProduction.parent)
        q = q.order_by('count desc')
        res = q.first()
        return res.count if res else 0

        # attr_idx = str(column).find('.')+1
        # attr_name = str(column)[attr_idx:]

        # # q_sum = sum([row.count for row in q])

        # for row in q:
        #     if getattr(row, attr_name) == column_value:
        #         return row.count#/float(q_sum)

        # return 0
        # q_probs = [float(row.count)/q_sum for row in q]
        # q_prods = [(row.lhs,row.rhs,row.parent,getattr(row, attr_name)) for row in q]
        # return zip(q_probs, q_prods)



    @classmethod
    def get_entropy_ratio_full_context(cls,
                                       lhs,
                                       rhs,
                                       column,
                                       parent=None,
                                       lmk_class=None,
                                       lmk_ori_rels=None,
                                       lmk_color=None,
                                       rel=None,
                                       dist_class=None,
                                       deg_class=None,
                                       golden=False,
                                       verbose=False):
        if verbose: print 'Calculating entropy ratio for %s column' % column

        # column entropy unconditinal
        z = cls.query(CProduction.lhs,
                      CProduction.rhs,
                      column,
                      func.sum(CProduction.count).label('count'),
                      golden=golden) \
                .filter(column!=None) \
                .filter(column!=',,,') \
                .group_by(column) \
                .all()

        max_entropy = 0
        max_sum = sum([prod.count for prod in z])
        if verbose: print 'total count (unconditional):', max_sum

        for prod in z:
            p = prod.count / max_sum
            max_entropy -= p * math.log(p)
            if verbose: print prod

        if verbose:
            print 'column entropy (unconditional):', max_entropy
            print '*'*70

        # column entropy conditioned on lhs and rhs
        q = cls.query(CProduction.lhs,
                      CProduction.rhs,
                      CProduction.parent,
                      CProduction.landmark_class,
                      CProduction.landmark_color,
                      CProduction.landmark_orientation_relations,
                      CProduction.relation,
                      CProduction.relation_degree_class,
                      CProduction.relation_distance_class,
                      column,
                      func.sum(CProduction.count).label('count'),
                      golden=golden) \
                .filter(CProduction.lhs==lhs) \
                .filter(column!=None) \
                .filter(column!=',,,') \
                .group_by(column, CProduction.parent)

        if rhs is not None: q = q.filter(CProduction.rhs==rhs)
        if parent is not None: q = q.filter(CProduction.parent==parent)
        if lmk_class is not None: q = q.filter(CProduction.landmark_class==lmk_class)
        if lmk_color is not None: q = q.filter(CProduction.landmark_color==lmk_color)
        if lmk_ori_rels is not None: q = q.filter(CProduction.landmark_orientation_relations==lmk_ori_rels)
        if rel is not None: q = q.filter(CProduction.relation==rel)
        if deg_class is not None: q = q.filter(CProduction.relation_degree_class==deg_class)
        if dist_class is not None: q = q.filter(CProduction.relation_distance_class==dist_class)

        q.all()

        cond_entropy = 0
        cond_sum = sum([res.count for res in q])
        if verbose: print 'total count (conditioned on lhs: %s, rhs: %s and parent: %s):' % (lhs,rhs,parent), cond_sum

        for prod in q:
            p = prod.count / cond_sum
            cond_entropy -= p * math.log(p)
            if verbose: print prod

        ratio = cond_entropy / max_entropy
        if cond_sum == 0: ratio = max_entropy

        if verbose:
            print 'column entropy (conditioned on lhs: %s, rhs: %s and parent: %s):' % (lhs,rhs,parent), cond_entropy
            print 'RATIO:', ratio

        return ratio

    @classmethod
    def get_unique_productions(cls, group_by_rhs=False, group_by_parent=False, golden=False):
        groupby = [CProduction.lhs]
        if group_by_rhs: groupby.append(CProduction.rhs)
        if group_by_parent: groupby.append(CProduction.parent)

        q = cls.query(golden=golden).group_by(*groupby)

        return q.all()


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
                                 deg_class=None,
                                 golden=False,
                                 multiply=False):

        # if multiply:
        #     cp_db = cls.get_production_counts(lhs=lhs,
        #                                       parent=parent,
        #                                       lmk_class=lmk_class,
        #                                       lmk_ori_rels=lmk_ori_rels,
        #                                       lmk_color=lmk_color,
        #                                       rel=rel,
        #                                       dist_class=dist_class,
        #                                       deg_class=deg_class,
        #                                       golden=golden)
        #     if cp_db.count() <= 0:
        #         update *= 10
        #     else:
        #         ccounter = defaultdict(int)
        #         ccounter[rhs] = 0
        #         for cprod in cp_db.all():
        #             ccounter[cprod.rhs] += cprod.count

        #         ckeys, ccounts = zip(*ccounter.items())
        #         ccounts = np.array(ccounts, dtype=float)
        #         total = ccounts.sum()
        #         update *= total

        committed = False
        while not committed:

            try:
                cp_db = cls.get_production_counts(lhs=lhs,
                                                  rhs=rhs,
                                                  parent=parent,
                                                  lmk_class=lmk_class,
                                                  lmk_ori_rels=lmk_ori_rels,
                                                  lmk_color=lmk_color,
                                                  rel=rel,
                                                  dist_class=dist_class,
                                                  deg_class=deg_class,
                                                  golden=golden)

                num_results = cp_db.count()
                if num_results <= 0:
                    if update > 0:
                        # logger( 'Updating by %f, %f' % (update, update), 'warning')
                        count = update
                        CProduction(lhs=lhs,
                                    rhs=rhs,
                                    parent=parent,
                                    landmark_class=lmk_class,
                                    landmark_orientation_relations=lmk_ori_rels,
                                    landmark_color=lmk_color,
                                    relation=rel,
                                    relation_distance_class=dist_class,
                                    relation_degree_class=deg_class,
                                    count=count)

                # elif num_results == 1:

                #     cprod = cp_db.one()
                #     if multiply:
                #         # logger( 'Updating by %f, %f' % (update, ups[cprod.rhs]), 'warning')
                #         cprod.count *= 1+update
                #         if cprod.count < 1: cprod.count = 1
                #     else:
                #         # logger( 'Updating by %f, %f' % (update, ups[cprod.rhs]), 'warning')
                #         if cprod.count <= -update: cprod.count = 1
                #         else: cprod.count += update

                else:

                    ccounter = {}
                    for cprod in cp_db.all():
                        # print cprod.rhs, cprod.count
                        if cprod.rhs in ccounter: ccounter[cprod.rhs] += cprod.count
                        else: ccounter[cprod.rhs] = cprod.count

                    # print '----------------'

                    ckeys, ccounts = zip(*ccounter.items())

                    ccounts = np.array(ccounts, dtype=float)
                    # print 'models.py:559', ccounts
                    ccounts /= ccounts.sum()
                    updates = ccounts * update
                    ups = dict( zip(ckeys, updates) )

                    if multiply:
                        for cprod in cp_db.all():
                            # logger( 'Updating by %f, %f' % (update, ups[cprod.rhs]), 'warning')
                            assert( not np.isnan( ups[cprod.rhs] ) )
                            cprod.count *= 1+ups[cprod.rhs]
                            if cprod.count < 1: cprod.count = 1
                    else:
                        for cprod in cp_db.all():
                            # logger( 'Updating by %f, %f' % (update, ups[cprod.rhs]), 'warning')
                            if cprod.count <= -ups[cprod.rhs]: cprod.count = 1
                            else: cprod.count += ups[cprod.rhs]


                session().commit()
                committed = True
            except Exception as e:
                logger( 'Could not commit', 'warning' )
                logger( e )
                session().rollback()
                continue

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
        return cls.query().filter_by(word=word, **given).one()

    @classmethod
    def probability(cls, word, **given):
        try:
            wp = cls.get_prob(word=word, **given)
        except:
            wp = cls.calc_prob(word=word, **given)
            session().commit()
        return wp.count / wp.all_count

    @classmethod
    def update(cls, word, update_by, **given):
        try:
            wp = cls.get_prob(word=word, **given)
        except:
            wp = cls.calc_prob(word=word, **given)
        wp.all_count = wp.all_count + update_by
        wp.count = wp.count + update_by
        session().commit()
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
        return cls.query().filter_by(rhs=rhs, **params).one()

    @classmethod
    def probability(cls, rhs, **given):
        try:
            ep = cls.get_prob(rhs=rhs, **given)
        except:
            ep = cls.calc_prob(rhs=rhs, **given)
            session().commit()
        return ep.count / ep.all_count

    @classmethod
    def update(cls, rhs, update_by, **given):
        try:
            ep = cls.get_prob(rhs=rhs, **given)
        except:
            ep = cls.calc_prob(rhs=rhs, **given)
        ep.all_count = ep.all_count + update_by
        ep.count = ep.count + update_by
        session().commit()
        return


class SentenceParse(Base):
    id = Column(Integer, primary_key=True)

    sentence = Column(String)
    original_parse = Column(String)
    modified_parse = Column(String)

    @classmethod
    def get_sentence_parse(cls, sentence, orig_parse=None, mod_parse=None):
        q = cls.query()
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
            session().commit()

    @classmethod
    def add_sentence_parse_blind(cls, sentence, orig_parse, mod_parse):
        SentenceParse(sentence=sentence,
                      original_parse=orig_parse,
                      modified_parse=mod_parse)



if __name__ == '__main__':
    engine.echo = True
    create_all()
