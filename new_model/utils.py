#!/usr/bin/env python

from __future__ import division

import sys
# import random
sys.path.insert(1,"..")
from myrandom import random
from functools import partial
import inspect
from collections import defaultdict

import numpy as np
from planar import Vec2, BoundingBox

# import stuff from semantics
sys.path.append('..')
from semantics.speaker import Speaker
from semantics.landmark import Landmark, ObjectClass
from semantics.representation import RectangleRepresentation, PointRepresentation
from semantics.scene import Scene
from semantics.relation import OrientationRelationSet
from semantics.run import construct_training_scene

NONTERMINALS = ('LOCATION-PHRASE', 'RELATION', 'LANDMARK-PHRASE', 'LANDMARK')

class printcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

    map = {'header':HEADER,
           'okblue':OKBLUE,
           'okgreen':OKGREEN,
           'warning':WARNING,
           'fail':FAIL,
           'endc':ENDC}

def get_lmk_ori_rels_str(lmk):
    rel_str = ( ','.join([rel.__name__ if rel.__name__ in [r.__name__ for r in lmk.ori_relations] else '' for rel in OrientationRelationSet.relations]) ) if lmk else None
    #if lmk: print 'getting rel str', OrientationRelationSet.relations, lmk, lmk.ori_relations, rel_str
    return rel_str

def parent_landmark(lmk):
    """gets a landmark and returns its parent landmark
    or None if it doesn't have one"""
    if lmk:
        parent = lmk.parent
        if parent and not isinstance(parent, Landmark):
            parent = parent.parent_landmark
        return parent

def count_lmk_phrases(t):
    """gets an nltk Tree and returns the number of LANDMARK-PHRASE nodes"""
    return sum(1 for n in t.subtrees() if 'LANDMARK-PHRASE' in n.node)



# a wrapper for a semantics scene
class ModelScene(object):
    def __init__(self, scene=None, speaker=None):
        self.scene = scene
        if scene is None:
            self.scene = Scene(3)

            # not a very furnished scene, we only have one table
            table = Landmark('table',
                             RectangleRepresentation(rect=BoundingBox([Vec2(5,5), Vec2(6,7)])),
                             None,
                             ObjectClass.TABLE)

            self.scene.add_landmark(table)
            self.table = table
        self.set_scene(scene,speaker)

    def set_scene(self,scene,speaker):
        self.scene = scene
        self.speaker = speaker
        self.table = self.scene.landmarks['table']

        # there is a person standing at this location
        # he will be our reference
        if speaker is None:
            self.speaker = Speaker(Vec2(5.5, 4.5))
        else:
            self.speaker = speaker

        # NOTE we need to keep around the list of landmarks so that we can
        # access them by id, which is the index of the landmark in this list
        # collect all possible landmarks
        self.landmarks = []
        for scene_lmk in self.scene.landmarks.itervalues():
            self.landmarks.append(scene_lmk)

            # a scene can be represented as a plane, line, etc
            # each representation of a scene has different landmarks
            rs = [scene_lmk.representation]
            rs.extend(scene_lmk.representation.get_alt_representations())

            for r in rs:
                for lmk in r.get_landmarks():
                    self.landmarks.append(lmk)

        # FIXME we are using sentences with 1 or 2 LANDMARK-PHRASEs
        # so we need to restrict the landmarks to 0 or 1 ancestors
        self.landmarks = [l for l in self.landmarks if l.get_ancestor_count() < 2]

    def get_rand_loc(self):
        """returns a random location on the table"""
        bb = self.table.representation.get_geometry()
        xmin, ymin = bb.min_point
        xmax, ymax = bb.max_point
        return random.uniform(xmin, xmax), random.uniform(ymin, ymax)

    def get_landmark_id(self, lmk):
        return self.landmarks.index(lmk)

    def get_landmark_by_id(self, lmk_id):
        return self.landmarks[lmk_id]

    def sample_lmk_rel(self, loc, num_ancestors=None, usebest=False):
        """gets a location and returns a landmark and a relation
        that can be used to describe the given location"""
        landmarks = self.landmarks

        if num_ancestors is not None:
            landmarks = [l for l in landmarks if l.get_ancestor_count() == num_ancestors]

        loc = Landmark(None, PointRepresentation(loc), None, None)
        lmk, lmk_prob, lmk_entropy, head_on = self.speaker.sample_landmark( landmarks, loc, usebest=usebest)
        rel, rel_prob, rel_entropy = self.speaker.sample_relation(loc, self.table.representation.get_geometry(), head_on, lmk, step=0.5, usebest=usebest)
        rel = rel(head_on,lmk,loc)

        return (lmk, lmk_prob, lmk_entropy), (rel, rel_prob, rel_entropy)


# we will use this instance of the scene
scene = ModelScene( *construct_training_scene() )



# helper functions
def lmk_id(lmk):
    if lmk: return scene.get_landmark_id(lmk)

def rel_type(rel):
    if rel: return rel.__class__.__name__

def get_meaning(loc=None, num_ancestors=None, usebest=False):
    if not loc:
        loc = scene.get_rand_loc()

    lmk, rel = scene.sample_lmk_rel(Vec2(*loc), num_ancestors, usebest=usebest)
    # print 'landmark: %s (%s)' % (lmk, lmk_id(lmk))
    # print 'relation:', rel_type(rel)
    return lmk, rel

def m2s(lmk, rel):
    """returns a string that describes the gives landmark and relation"""
    return '<lmk=%s(%s, %s), rel=%s(%s,%s)>' % (repr(lmk), lmk_id(lmk), lmk.object_class if lmk else None, rel_type(rel),
                                                rel.measurement.best_degree_class if hasattr(rel,'measurement') else None,
                                                rel.measurement.best_distance_class if hasattr(rel,'measurement') else None)

def laplace_estimator(f, N, n):
    return (N * f + 1) / float( N + n)

def shannon_entropy_of_counts(counts, N=None, worst_counts=None):
    counts = np.array(counts, dtype=float)
    probs = counts / counts.sum()
    if worst_counts is not None:
        worst_counts = np.array(worst_counts, dtype=float)
        worst_probs = worst_counts / worst_counts.sum()
    else:
        worst_probs = None
    return shannon_entropy_of_probs(probs, N=N, worst_probs=worst_probs)

def shannon_entropy_of_probs(probs, N=None, worst_probs=None):
    probs = np.array(probs)
    if N is not None:
        probs = laplace_estimator(np.array(probs),N,len(probs))
    if worst_probs is not None:
        probs = probs*np.array(worst_probs)
        probs = probs/probs.sum()
    temp = (probs * np.log(probs))
    temp[probs==0.0] = 0.0
    return -np.sum( temp )

def zrm_entropy_of_counts(counts, worst_counts, N=None):
    counts = np.array(counts, dtype=float)
    probs = counts / counts.sum()
    worst_counts = np.array(worst_counts, dtype=float)
    worst_probs = worst_counts / worst_counts.sum()
    return zrm_entropy(probs, worst_probs, N)


def zrm_entropy(ps, worst_ps, N=None):
    if N is not None:
        nom = [laplace_estimator( f, N, len(ps) ) * ( 1 - laplace_estimator( f, N, len(ps) ) )  for f in ps]
        denom = [ (-2 * w + 1) * laplace_estimator(w, N, len(worst_ps) ) + w * 2 for w in worst_ps]
    else:
        nom = [f * ( 1 - f )  for f in ps]
        denom = [ (-2 * w + 1) * w + w * 2 for w in worst_ps]
    return sum([ni / denomi for ni,denomi in zip(nom, denom)])

def min_entropy(probs, N=None, worst_probs=None):
    if N is not None:
        probs = laplace_estimator(np.array(probs),N,len(probs))
    if worst_probs is not None:
        probs = probs*np.array(worst_probs)
        probs = probs/probs.sum()
    return -np.log(max(probs))



def categorical_sample(values, probs):
    probs = np.array(probs)
    index = np.random.multinomial(1, probs).nonzero()[0][0]
    value = values[index]
    return value, probs[index], -np.sum( (probs * np.log(probs)) )

def pick_best(values, probs):
    prob, value = max(zip(probs,values))
    return value, prob, -np.sum( (probs * np.log(probs)) )


# based on matlab's mvnpdf
def mvnpdf(x, mu=0, sigma=None):
    """multivariate normal probability density function"""

    # number of elements in `x`
    k = len(x)

    # `x` must be a numpy array
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    # if `mu` is a scalar value then convert it into a numpy array
    # with the scalar value repeated `k` times
    if isinstance(mu, (int, float)):
        mu *= np.ones(k)

    # `mu` must be a numpy array
    elif not isinstance(mu, np.ndarray):
        mu = np.array(mu)

    # if `sigma` is unspecified then create a k*k identity matrix
    if sigma is None:
        sigma = np.identity(k)

    # make sure `sigma` is a numpy array
    elif not isinstance(sigma, np.ndarray):
        sigma = np.array(sigma)

    # if `sigma` is a 1d array then convert it into a matrix
    # with the vector in the diagonal
    if sigma.ndim == 1:
        sigma = np.diag(sigma)

    # difference between `x` and the mean
    diff = x - mu

    # calculate probability density
    pd = (2*np.pi) ** (-k/2)
    pd *= np.linalg.det(sigma) ** (-1/2)
    pd *= np.exp(-1/2 * diff.dot(np.linalg.inv(sigma)).dot(diff))

    return pd



def force_unicode(s, encoding='utf-8', errors='strict'):
    """convert to unicode or die trying"""
    if isinstance(s, unicode):
        return s
    elif isinstance(s, str):
        return s.decode(encoding, errors)
    elif hasattr(s, '__unicode__'):
        return unicode(s)
    else:
        return str(s).decode(encoding, errors)


def logger(msg, color=''):
    color = printcolors.map[color] if color in printcolors.map else ''
    end = '' if color == '' else ''
    fn, line = inspect.stack()[1][1:3]
    fn = fn[fn.rfind('/')+1:]
    print "%s%s:%d - %s%s" % (color, fn, line, msg, printcolors.ENDC)


# generates a list of tuples of size `n`
# each tuple is an ngram and includes the right number
# of start and end tokens ('<s>' and '</s>' by default)
# `tokens` is a list of tokens which could be strings (words)
# or objects like `models.Word`
def ngrams(tokens, n, start_tk='<s>', end_tk='</s>'):
    tokens = [start_tk] * (n-1) + tokens + [end_tk] * (n>1)
    return [tuple(tokens[i:i+n]) for i in xrange(len(tokens)-n+1)]

bigrams = partial(ngrams, n=2)
trigrams = partial(ngrams, n=3)
