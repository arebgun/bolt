import collections as coll
import numpy as np

def indent(lines, amount=2, ch=' '):
    padding = amount * ch
    return '\n'.join([padding + line for line in lines.split('\n')[:-1]])+'\n'

class Applicabilities(coll.defaultdict):
    def __init__(self, pairs):
        super(Applicabilities, self).__init__(float)
        self.update(pairs)

    def __mul__(self, other):
        keys = set(self.keys()+other.keys())
        # product = Applicabilities([(key,self[key]*other[key]) for key in keys])
        pairs = []
        for key in keys:
            if np.isnan(self[key]) or np.isnan(other[key]):
                pairs.append((key, 0.0))
            else:
                pairs.append((key, self[key]*other[key]))
        product = Applicabilities(pairs)
        return product

class Context(object):

    def __init__(self, scene, speaker):
        self.scene = scene
        self.speaker = speaker
        self.entities = scene.landmarks.values()
        self.potential_referents = [(entity,) for entity in self.entities]
        self.all_potential_referents = [(entity,) for entity in self.entities]
        self.all_potential_referents += [(landmark,) for landmark in 
                    scene.landmarks['table'].representation.get_landmarks(1)]
        # for r in range(2, min(len(self.entities),2)+1):
        #     self.potential_referents.extend(it.permutations(self.entities,r))

    def get_entities(self):
        return list(self.entities)

    def get_potential_referents(self):
        return list(self.potential_referents)

    def get_all_potential_referents(self):
        return list(self.all_potential_referents)

    def get_potential_referent_scores(self):
        pairs = zip(self.potential_referents,[1]*len(self.potential_referents))
        potential_referent_scores = Applicabilities(pairs)
        return potential_referent_scores

    def get_all_potential_referent_scores(self):
        pairs = zip(self.all_potential_referents,[1]*len(self.all_potential_referents))
        potential_referent_scores = Applicabilities(pairs)
        return potential_referent_scores

class Match(object):
    def __init__(self, start, end, construction, constituents):
        self.start = start
        self.end = end
        self.construction = construction
        self.constituents = constituents
        self.num_holes = 0
        self.hole_width = 0
        if constituents:
            for c in constituents:
                if isinstance(c, Hole):
                    self.num_holes += 1
                    self.hole_width += len(c.unmatched_sequence)
       
    def __repr__(self):
        return '<%s %s>'%(self.construction,(self.start,self.end))

    # def find_partials(self):
    #     return [self]

    # def get_holes(self):
    #     return [c for c in self.constituents if isinstance(c, Hole)]

    def prettyprint(self):
        string = 'Partial '+self.construction.__name__+'\n'
        for constituent in self.constituents:
            string += indent(constituent.prettyprint())
        return string

class Hole(object):
    def __init__(self, unmatched_pattern, unmatched_sequence):
        self.unmatched_pattern = unmatched_pattern
        self.unmatched_sequence = unmatched_sequence

    def __repr__(self):
        return '<Hole %s %s>'%(self.unmatched_pattern, self.unmatched_sequence)
        
    def sempole(self):
        return self._sempole.copy()

    def equivalence(self, other):
        if other == None:
            return -2
        else:
            return other.equivalence(self)

    def collect_leaves(self):
        leaves = []
        for constituent in self.unmatched_sequence:
            leaves.extend(constituent.collect_leaves())
        return leaves

    def print_sentence(self):
        leaves = self.collect_leaves()
        return ' '.join(leaves)

    def prettyprint(self):
        string = self.__class__.__name__+' %s\n'%self.unmatched_pattern
        for item in self.unmatched_sequence:
            string += indent(item.prettyprint())
        return string


def tuple_extend(tup, to_extend):
    return tuple(list(tup)+to_extend)