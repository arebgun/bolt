#!/usr/bin/python
import sys
sys.path.insert(1, '..')
from automain import automain
from argparse import ArgumentParser
from lexical_items import Space, lexical_items_list
from constructions import constructions_list

def recursive_parse(partial_parse):
    # print partial_parse
    if len(partial_parse) == 1:
        return [partial_parse]
    matches = []
    for c in constructions_list:
        for m in c.match(partial_parse):
            matches.append((m,c))
            # print '  ',m, c
    parses = []
    for (start, end), construction in matches:
        new_partial_parse = partial_parse[:start]+\
                            [construction(partial_parse[start:end])]+\
                            partial_parse[end:]
        parses.extend(recursive_parse(new_partial_parse))
    return parses

def parse_sentence(sentence):
    all_matches = []
    for tc in lexical_items_list:
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
def test():
    argparser = ArgumentParser()
    argparser.add_argument('-t', '--text', type=str)
    args = argparser.parse_args()

    parses = parse_sentence(args.text)
    for parse in parses:
        print parse.prettyprint()