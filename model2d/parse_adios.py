#!/usr/bin/env python

import sys
sys.path.append('../../context-clustering')

from grammar import parse_pcfg
from pchart import LongestChartParser
from grammar_metrics import get_stats

from models import SentenceParse

#GRAMMAR_BASE_NAME = 'multi_scene_sentences_collapsed'
GRAMMAR_BASE_NAME = 'adios_corpus_c_l_collapsed'

class ParseError(RuntimeError):
    pass

def get_parser(write_grammar_stats=False):
    """
    Given a grammar induced by ADIOS algorithm construct a
    corresponding parser.
    """
    with open('grammars/%s.grammar' % GRAMMAR_BASE_NAME) as f:
        grammar = parse_pcfg(f, includes_counts=True)

    with open('grammars/%s.stats' % GRAMMAR_BASE_NAME, 'w') as f:
        f.write(get_stats(grammar))

    return LongestChartParser(grammar)

parser = get_parser()

def parse(sentence):
    """returns a sentence parse"""
    sp_db = SentenceParse.get_sentence_parse(sentence)

    try:
        res = sp_db.all()[0]
        parsetree = res.original_parse
    except:
        try:
            parses = parser.nbest_parse(sentence.split(), n=5)[0]
            if len(parses) == 0: raise ParseError('Parser was unable to generate a valid parse.')
            parsetree = parses[0].pprint()
            SentenceParse.add_sentence_parse(sentence, parsetree, parsetree)
        except Exception as e:
            raise e

    return parsetree

if __name__ == '__main__':
    normal_parser = get_parser()
