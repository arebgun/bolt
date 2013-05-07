#!/usr/bin/env python
# coding: utf-8
import os
import csv
import sys
import tempfile
import fileinput
import subprocess
from models import SentenceParse
from utils import count_lmk_phrases, printcolors
from nltk.tree import ParentedTree


class ParseError(RuntimeError):
    pass

def is_nonterminal(s):
    return s.upper() == s

def modify_parses(trees, tregex_path='stanford-tregex', surgery_path='surgery'):
    """modify parse trees using tsurgeon"""

    temp = tempfile.NamedTemporaryFile()
    temp.write('\n'.join(trees))
    temp.flush()

    # tregex jar location
    jar = os.path.join(tregex_path, 'stanford-tregex.jar')
    tsurgeon = 'edu.stanford.nlp.trees.tregex.tsurgeon.Tsurgeon'

    # surgery scripts
    surgery = [os.path.join(surgery_path, 'remove_frag.ts')]
    proc = subprocess.Popen(['java', '-mx100m', '-cp', jar, tsurgeon,
                             '-s', '-treeFile', temp.name] + surgery,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    output = proc.communicate()[0]
    temp.close()

    return map(ParentedTree.parse, output.splitlines())

def parse_sentences(ss, parser_path='../bllip-parser', n=5, threads=2):
    """parse sentences with the charniak parser"""

    # create a temporary file and write the sentences in it
    temp = tempfile.NamedTemporaryFile()

    for s in ss:
        temp.write('<s> %s </s>\n' % s)

    temp.flush()

    # where am i?
    # prev_path = os.getcwd()
    # get into the charniak parser directory
    # os.chdir(parser_path)

    # call the parser
    proc = subprocess.Popen(['./parse.sh', '-N%i' % n, temp.name],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    output = proc.communicate()[0]
    temp.close()

    # return to where i was
    # os.chdir(prev_path)

    # ParentedTree.parse(modparsetree)
    parses = modify_parses([l for l in output.splitlines() if l[:3] == '(S1'])

    return parses

def parse(sentence):
    """returns a sentence parse"""
    sp_db = SentenceParse.get_sentence_parse(sentence)

    try:
        res = sp_db.all()[0]
        parsetree = res.original_parse
    except:
        try:
            temp = tempfile.NamedTemporaryFile()
            temp.write('<s> %s </s>\n' % sentence)
            temp.flush()
            proc = subprocess.Popen(['./parse.sh', '-N%i' % 1, temp.name],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

            output = proc.communicate()[0]
            temp.close()
            parses = modify_parses([l for l in output.splitlines() if l[:3] == '(S1'])

            if len(parses) == 0: raise ParseError('Parser was unable to generate a valid parse.')
            parsetree = parses[0].pprint()
            SentenceParse.add_sentence_parse(sentence, parsetree, parsetree)
        except Exception as e:
            raise e

    return parsetree
