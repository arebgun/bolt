#!/usr/bin/python
import sys
sys.path.insert(1,'..')
from automain import automain
import itertools as it
import collections as coll
import IPython
import re

class Entity(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

class Context(object):

    def __init__(self, entities):
        self.entities = entities
        self.potential_referents = [(entity,) for entity in self.entities]
        for r in range(2, min(len(self.entities),2)+1):
            self.potential_referents.extend(it.permutations(self.entities,r))

    def get_potential_referents(self):
        return self.potential_referents

    def get_potential_referent_scores(self):
        return dict(zip(self.potential_referents,[1]*len(self.potential_referents)))

class Property(object):
    pass

class Gradable(Property):
    pass

class Applicable(Property):
    def applicabilities(self, context):
        potential_referent_scores = context.get_potential_referent_scores()
        return potential_referent_scores


class TextConstruction(object):
    def __init__(self, regex):
        self.regex = regex

    def match(self, text):
        matches = re.finditer(self.regex, text)
        intervals = [(m.start(),m.end()) for m in matches]
        return intervals

    def __repr__(self):
        return self.__class__.__name__ + ' "' + self.regex + '"'

    def prettyprint(self):
        return self.__repr__() + '\n'

    def __hash__(self):
        return hash(self.prettyprint())

    def __eq__(self, other):
        return hash(self) == hash(other)

class Space(TextConstruction):
    pass

class Article(TextConstruction):
    def __init__(self, regex, cardinality, unique):
        super(Article, self).__init__(regex)
        self.cardinality = cardinality
        self.unique = unique

class Noun(TextConstruction, Applicable):
    pass

class Plural(TextConstruction):
    pass

class Adjective(TextConstruction, Applicable):
    pass

class Degree(TextConstruction):
    pass

def indent(lines, amount=2, ch=' '):
    padding = amount * ch
    return '\n'.join([padding + line for line in lines.split('\n')[:-1]])+'\n'

class Construction(object):
    
    def __init__(self, constituents):
        assert(len(constituents) == len(self.pattern))
        self.constituents = constituents

    @classmethod
    def match(cls, sequence):
        matches = []
        patlength = len(cls.pattern)
        for i, item in enumerate(sequence[:len(sequence)-patlength+1]):
            if isinstance(item, cls.pattern[0]):
                if len(cls.pattern) == len(sequence[i:i+patlength]):
                    all_match = True
                    for pat, seq in zip(cls.pattern, sequence[i:i+patlength]):
                        # print seq, pat, isinstance(seq, pat)
                        all_match = all_match and isinstance(seq, pat)
                    if all_match:
                        matches.append((i,i+patlength))
        return matches

    def __repr__(self):
        return self.__class__.__name__

    def prettyprint(self):
        string = self.__repr__()+'\n'
        for constituent in self.constituents:
            string += indent(constituent.prettyprint())
        return string

    def __hash__(self):
        return hash(self.prettyprint())

    def __eq__(self, other):
        return hash(self) == hash(other)

class GradablePhrase(Construction):
    pass

class Relation(Property):
    def applicabilities(self, context, landmark):
        return self.constituents[0].applicabilities(context, landmark)

class SimpleSpatialRelation(TextConstruction, Relation):
    pass

class CompoundSpatialRelation(Construction, Relation):
    pass

class Direction(TextConstruction):
    pass

class SourceTag(TextConstruction):
    pass

class DestinationTag(TextConstruction):
    pass

class BelongingTag(TextConstruction):
    pass

class Measure(Property):
    pass

class GradableMeasure(TextConstruction, Measure, Gradable):
    pass

# class Number(TextConstruction):
#     pass

# class Unit(TextConstruction):
#     pass

# class ConcreteMeasure(Measure):
#     pass

class DistantMeasure(GradableMeasure):
    pass

class ProximitousMeasure(GradableMeasure):
    pass

class DistantMeasurePhrase(Construction):
    pass

class PlainDistantMeasurePhrase(DistantMeasurePhrase):
    pattern = [DistantMeasure]

class DegreeDistantMeasurePhrase(DistantMeasurePhrase):
    pattern = [Degree, DistantMeasure]

class ProximitousMeasurePhrase(Construction):
    pass

class PlainProximitousMeasurePhrase(ProximitousMeasurePhrase):
    pattern = [ProximitousMeasure]

class DegreeProximitousMeasurePhrase(ProximitousMeasurePhrase):
    pattern = [Degree, ProximitousMeasure]

class DistantRelation(CompoundSpatialRelation):
    pattern = [DistantMeasurePhrase, SourceTag]

class ProximitousRelation(CompoundSpatialRelation):
    pattern = [ProximitousMeasurePhrase, DestinationTag]



class OrientationRelation(CompoundSpatialRelation):
    pattern = [DestinationTag, Article, Direction, BelongingTag]



class AdjectivePhrase(Construction, Applicable):
    pass

class PlainAdjectivePhrase(AdjectivePhrase):
    pattern = [Adjective]

class DegreeAdjectivePhrase(AdjectivePhrase):
    pattern = [Degree, Adjective]

class NounPhrase(Construction, Applicable):
    pass

class PlainNounPhrase(NounPhrase):
    pattern = [Noun]

    def applicabilities(self, context):
        return self.constituents[0].applicabilities(context)

class AdjectiveNounPhrase(NounPhrase):
    pattern = [AdjectivePhrase, Noun]

    def applicabilities(self, context):
        noun_applicabilities = self.constituents[1].applicabilities(context)
        adjective_applicabilities = self.constituents[1].applicabilities(context)
        for referent in noun_applicabilities:
            noun_applicabilities[referent] *= adjective_applicabilities[referent]
        return noun_applicabilities
               

def cardinality_one(ref):
    return len(ref)==1

def cardinality_greater_than_one(ref):
    return len(ref)>=1

def cardinality_two(ref):
    return len(ref)==2

def cardinality_anything(ref):
    return True

class ReferringExpression(Construction, Applicable):

    def score_referents(self, context):
        context_score_dict = context.get_potential_referent_scores()
        
        applicability_dict = self.applicabilities(context)

        for referent in context_score_dict:
            if self.constituents[0].cardinality(referent):
                context_score_dict[referent] *= applicability_dict[referent]
            else: context_score_dict[referent] = 0

        return context_score_dict

    def choose_referent(self, context):
        score_dict = self.score_referents(context)
        potential_referents, scores = zip(*score_dict.items())

        score_tuples = sorted(zip(scores, potential_referents), reverse=True)
        best_score, best_referent = score_tuples[0]

        return best_referent

class IntrinsicReferringExpression(ReferringExpression):
    pattern = [Article, NounPhrase]

    def applicabilities(self, context):
        applicabilities = self.constituents[1].applicabilities(context)
        for referent in applicabilities:
            if not self.constituents[0].cardinality(referent):
                applicabilities[referent] = 0
        return applicabilities

class ExtrinsicReferringExpression(ReferringExpression):
    pattern = [Article, NounPhrase, Relation, ReferringExpression]

    def applicabilities(self, context):
        applicabilities = self.constituents[1].applicabilities(context)
        for referent in applicabilities:
            if not self.constituents[0].cardinality(referent):
                applicabilities[referent] = 0

        landmark_score_dict = self.constituents[3].score_referents(context)

        relation_expectations = coll.defaultdict(int)
        for landmark, landmark_score in landmark_score_dict.items():
            relation_apps = self.constituents[2].applicabilities(context, 
                                                            landmark=landmark)
            for potential_trajector in relation_apps:
                to_add = relation_apps[potential_trajector] * landmark_score
                relation_expectations[potential_trajector] += to_add
            
        for potential_trajector in applicabilities:
            applicabilities[potential_trajector] #*= \
            relation_expectations[potential_trajector]

        return applicabilities

_         = Space(regex=' ')
a         = Article(regex='a', cardinality=cardinality_one, unique=False) # num=1, unique=maybe not, known=probably not
some      = Article(regex='some', cardinality=cardinality_greater_than_one, unique=False) # num>1, unique=maybe not, known=probably not
the       = Article(regex='the', cardinality=cardinality_anything, unique=True) # num=?, unique=probably, known=probably

object__  = Noun(regex='object')
cube      = Noun(regex='cube')
block     = Noun(regex='block')
box       = Noun(regex='box')
sphere    = Noun(regex='sphere')
ball      = Noun(regex='ball')
cone      = Noun(regex='cone')
cylinder  = Noun(regex='cylinder')

_s        = Plural(regex='(?<!:\s)s(?=[\s",;.!?])')

big       = Adjective(regex='big')
large     = Adjective(regex='large')
small     = Adjective(regex='small')
little    = Adjective(regex='little')

red       = Adjective(regex='red')
orange    = Adjective(regex='orange')
yellow    = Adjective(regex='yellow')
blue      = Adjective(regex='blue')
purple    = Adjective(regex='purple')
black     = Adjective(regex='black')
white     = Adjective(regex='white')
gray      = Adjective(regex='gray')
grey      = Adjective(regex='grey')

very      = Degree(regex='very')
somewhat  = Degree(regex='somewhat')
pretty    = Degree(regex='pretty')
extremely = Degree(regex='extremely')

left      = Direction(regex='left')
right     = Direction(regex='right')
# north     = Direction(regex='north')
# south     = Direction(regex='south')
# east      = Direction(regex='east')
# west      = Direction(regex='west')

to        = DestinationTag(regex='to')
from__    = SourceTag(regex='from')
of        = BelongingTag(regex='of')

far       = DistantMeasure(regex='far')
near      = ProximitousMeasure(regex='near')

in_front_of = SimpleSpatialRelation(regex='in front of')
behind    = SimpleSpatialRelation(regex='behind')


text_constructions = [
    _,
    a,
    the,
    object__,
    cube,
    block,
    box,
    sphere,
    ball,
    cone,
    cylinder,
    _s,
    big,
    large,
    small,
    little,
    red,
    orange,
    yellow,
    blue,
    purple,
    black,
    white,
    gray,
    grey,
    very,
    somewhat,
    pretty,
    extremely,
    left,
    right,
    # north,
    # south,
    # east,
    # west,
    to,
    from__,
    of,
    far,
    near,
    in_front_of,
    behind,
]

constructions = [
    PlainAdjectivePhrase,
    DegreeAdjectivePhrase,
    PlainNounPhrase,
    AdjectiveNounPhrase,
    PlainDistantMeasurePhrase,
    DegreeDistantMeasurePhrase,
    PlainProximitousMeasurePhrase,
    DegreeProximitousMeasurePhrase,
    DistantRelation,
    ProximitousRelation,
    OrientationRelation,
    IntrinsicReferringExpression,
    ExtrinsicReferringExpression
]

def recursive_parse(partial_parse):
    if len(partial_parse) == 1:
        return [partial_parse]
    matches = []
    for c in constructions:
        for m in c.match(partial_parse):
            matches.append((m,c))
    parses = []
    for (start, end), construction in matches:
        new_partial_parse = partial_parse[:start]+\
                            [construction(partial_parse[start:end])]+\
                            partial_parse[end:]
        parses.extend(recursive_parse(new_partial_parse))
    return parses

def parse_sentence(sentence):
    all_matches = []
    for tc in text_constructions:
        for m in tc.match(sentence):
            all_matches.append((m,tc))
    # print all_matches

    partial_parses = []
    for match in all_matches:
        if match[0][0] == 0:
            partial_parses.append([match])
    # print partial_parses

    new_partial_parses = partial_parses
    something_new = True
    while something_new:
        partial_parses = new_partial_parses
        something_new = False
        new_partial_parses = []
        for partial_parse in partial_parses:
            previous_end = partial_parse[-1][0][1]
            for match in all_matches:
                if match[0][0] == previous_end:
                    new_parse = list(partial_parse)
                    new_parse.append(match)
                    new_partial_parses.append(new_parse)
                    something_new = True

    # print partial_parses

    for i, parse in enumerate(partial_parses):
        partial_parses[i] = [item[1] for item in parse if not isinstance(item[1],Space)]

    # print partial_parses

    partial_parse = partial_parses[0]

    parses = set()
    for partial_parse in partial_parses:
        new_parses = recursive_parse(partial_parse)
        new_parses = [p for (p,) in new_parses] #TODO check parse length?
        parses.update(new_parses)

    # print
    # for parse in parses:
    #     print parse.prettyprint()
    return parses

@automain
def main():
    
    sentences = [
        'the red sphere behind a blue block',
        'the object to the left of the yellow cylinder',
        'the cube near to the gray sphere',
        'the box very far from a purple cone'
    ]

    for sentence in sentences:
        print sentence
        parses = parse_sentence(sentence)

        for parse in parses:
            print parse.prettyprint()

    # the_only_parse = list(parses)[0]

    # names = ['red sphere',
    #      'blue block',
    #      'yellow cylinder']
    # context = Context([Entity(name) for name in names])

    # scores = the_only_parse.score_referents(context)
    # print scores

    # IPython.embed()