#!/usr/bin/python
import sys
sys.path.insert(1, '..')
import IPython
import argparse as ap
import operator as op

import utils
import automain
import common as cmn
import lexical_items as li
import constructions as con
import construction as struct
# global lexical_items_list
# global constructions_list



class Parse(object):
    def __init__(self, lexical_parse):
        self.parses = [lexical_parse]
        self.num_holes = 0
        self.hole_width = 0

    @property
    def current(self):
        return self.parses[-1]

    def update(self, match):
        new = self.copy()
        # print new
        s, e = match.start, match.end
        if match.num_holes == 0:
            new_construction = match.construction(match.constituents)
        else:
            new_construction = match
            new.num_holes += match.num_holes
            new.hole_width += match.hole_width

        updated_parse = new.current[:s]+\
                        [new_construction]+\
                        new.current[e:]
        new.parses.append(updated_parse)
        return new

    def copy(self):
        copy = object.__new__(self.__class__)
        copy.parses = list(self.parses)
        copy.num_holes = self.num_holes
        copy.hole_width = self.hole_width
        return copy


    def __str__(self):
        return '\n'.join(['%i %s'%(i,parse) 
                          for i,parse in enumerate(self.parses)])

    def __hash__(self):
        return hash('\n'.join(c.prettyprint() for c in self.current))

    def __eq__(self, other):
        return hash(self) == hash(other)


def get_all_lexical_parses(sentence, lexicon):
    all_matches = []
    # global lexical_items_list
    for tc in lexicon:
        all_matches.extend(tc.match(sentence))
    partial_parses = [[cmn.Match(0,0,li._,None)]]

    finished_parses = []
    new_partial_parses = partial_parses
    # something_new = True
    while new_partial_parses != []:
        partial_parses = new_partial_parses
        # something_new = False
        new_partial_parses = []
        for partial_parse in partial_parses:
            if partial_parse[-1].end == len(sentence):
                finished_parses.append(partial_parse)
                continue
            new_this_parse_list = []
            previous_end = partial_parse[-1].end
            for match in all_matches:
                if match.start == previous_end:
                    new_parse = list(partial_parse)
                    new_parse.append(match)
                    new_this_parse_list.append(new_parse)
                    # something_new = True
            if new_this_parse_list == []:
                # No immediate matches, find what we need to skip
                # something_else_new = True
                new_min_end = float('inf')
                next_matches = next_lowest_matches(previous_end, 
                                                   new_min_end,
                                                   all_matches)
                for next_match in next_matches:
                    new_parse = list(partial_parse)
                    new_parse.append(
                        cmn.Match(previous_end, 
                                  match.start, 
                                  struct.Unknown(
                                    sentence[previous_end:
                                             next_match.start]),
                                  None
                                  ))
                    new_parse.append(next_match)
                    new_this_parse_list.append(new_parse)
                if next_matches == []:
                    new_parse = list(partial_parse)
                    new_parse.append(
                        cmn.Match(previous_end, 
                                  len(sentence), 
                                  struct.Unknown(
                                    sentence[previous_end:]),
                                  None
                                  ))
                    new_this_parse_list.append(new_parse)
            new_partial_parses.extend(new_this_parse_list)

    return finished_parses

def next_lowest_matches(start, end, all_matches):
    matches = []
    lowest_end = float('inf')
    next_lowest = True
    while next_lowest:
        next_lowest = lowest_between(start, end, [], all_matches)
        if next_lowest is not None:
            matches.append(next_lowest)
            if next_lowest.end < lowest_end:
                end = next_lowest.end
    return matches

def lowest_between(start, end, already_found, all_matches):
    lowest_so_far = cmn.Match(end, end, None, None)
    for match in all_matches:
        if (match.start >= start 
                and match.start <= lowest_so_far.start
                and match.end < end 
                and match not in already_found):
            lowest_so_far = match
    if lowest_so_far.start == end:
        return None
    else:
        return lowest_so_far

def get_all_construction_parses(lexical_parses, structicon, max_holes=0):

    for i, parse in enumerate(lexical_parses):
        lexical_parses[i] = [match.construction for match in parse
                             if not isinstance(match.construction,li.Space)]

    unfinished_parses = [Parse(lp) for lp in lexical_parses]

    parses = set()
    for uparse in unfinished_parses:
        new_parses = recursive_parse(unfinished_parse=uparse, 
                                     structicon=structicon, 
                                     max_holes=max_holes)
        parses.update(new_parses)

    return parses

def recursive_parse(unfinished_parse, structicon, max_holes=0):
    if len(unfinished_parse.current) == 1:
        return [unfinished_parse]

    parses = set()
    # global constructions_list
    for c in structicon:
        cmatches = c.match(unfinished_parse.current)

        for match in cmatches:
            new_parse = unfinished_parse.update(match)
            parses.update(recursive_parse(unfinished_parse=new_parse, 
                                          structicon=structicon,
                                          max_holes=max_holes))
            
        else:
            if unfinished_parse.num_holes < max_holes:
                partial_matches = c.partially_match(unfinished_parse.current)
                for match in partial_matches:
                    new_parse = unfinished_parse.update(match)
                    finished_parses=recursive_parse(unfinished_parse=new_parse, 
                                                    structicon=structicon,
                                                    max_holes=max_holes)
                    parses.update(finished_parses)
    return parses


@automain.automain
def test():

    global lexical_items_list 
    lexical_items_list = [
        li._,
        # li.a,
        li.the,
        li.objct,
        # li.cube,
        li.block,
        # li.box,
        li.sphere,
        # li.ball,
        # li.cone,
        li.cylinder,
        # li._s,
        # li.big,
        # li.large,
        # li.small,
        # li.little,
        li.red,
        li.orange,
        li.yellow,
        li.green,
        # li.blue,
        # li.purple,
        # li.black,
        # li.white,
        # li.gray,
        # li.grey,
        # li.very,
        # li.somewhat,
        # li.pretty,
        # li.extremely,
        # li.front,
        # li.back,
        li.left,
        # li.right,
        # li.north,
        # li.south,
        # li.east,
        # li.west,
        li.to,
        li.frm,
        li.of,
        # li.far,
        # li.near,
    ]

    global constructions_list
    constructions_list = [
        con.AdjectivePhrase,
        con.DegreeAdjectivePhrase,
        con.NounPhrase,
        con.AdjectiveNounPhrase,
        con.MeasurePhrase,
        con.DegreeMeasurePhrase,
        con.DistanceRelation,
        con.OrientationRelation,
        con.ReferringExpression,
        con.RelationLandmarkPhrase,
        con.RelationNounPhrase,
        con.ExtrinsicReferringExpression
    ]

    argparser = ap.ArgumentParser()
    argparser.add_argument('-t', '--text', type=str)
    args = argparser.parse_args()

    lexical_parses = get_all_lexical_parses(args.text)
    for parse in lexical_parses:
        print parse
    print
    parses = get_all_construction_parses(lexical_parses, max_holes=1)
    print len(parses)
    parses = sorted(parses, key=op.attrgetter('hole_width'), reverse=True)
    for parse in parses:
        print 'parse length:', len(parse.current)
        print 'num_holes:', parse.num_holes, 'hole_width:', parse.hole_width
        print parse.current[0].prettyprint()