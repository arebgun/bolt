#!/usr/bin/python
import itertools as it
import collections as coll
from parser import parse_sentence

class Entity(object):
    def __init__(self, shape, color):
        self.concrete = True
        self.shape = shape
        self.color = color

    def __repr__(self):
        return 'shape: %s, color: %s' % (self.shape, self.color)

class Context(object):

    def __init__(self, entities):
        self.entities = entities
        self.potential_referents = [(entity,) for entity in self.entities]
        for r in range(2, min(len(self.entities),2)+1):
            self.potential_referents.extend(it.permutations(self.entities,r))

    def get_entities(self):
        return list(self.entities)

    def get_potential_referents(self):
        return list(self.potential_referents)

    def get_potential_referent_scores(self):
        potential_referent_scores = coll.defaultdict(float)
        potential_referent_scores.update(zip(self.potential_referents,[1]*len(self.potential_referents)))
        return potential_referent_scores

if __name__ == '__main__':

    # sentences = [
    #     'the red sphere behind a blue block',
    #     'the object to the left of the yellow cylinder',
    #     'the cube near to the gray sphere',
    #     'the box very far from a purple cone'
    # ]

    sentences = [
        'the red block',
        'the blue sphere',
        'the green cone',
        'the purple cylinder',
        'the purple block',
        'the green sphere',
        'the blue cone',
        'the red cylinder'
    ]

    for sentence in sentences:
        print sentence
        parses = parse_sentence(sentence)

        for parse in parses:
            print parse.prettyprint()

    entities = [
        Entity(shape='block', color='red'),
        Entity(shape='sphere', color='blue'),
        Entity(shape='cone', color='green'),
        Entity(shape='cylinder', color='purple')
    ]

    context = Context(entities=entities)

    # potential_referent_dict = context.get_potential_referent_scores()
    for parse in parses:
        print parse.applicabilities(context)