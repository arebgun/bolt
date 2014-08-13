#!/usr/bin/env python
# coding: utf-8

# import re
# import os
# import csv
# import sys
# import glob
import tempfile
# import fileinput
import subprocess
# import traceback
# from models import SentenceParse
# from sqlalchemy.orm.exc import NoResultFound
# from utils import count_lmk_phrases, printcolors
# from nltk.tree import ParentedTree

def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

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
    # capture output
    output = proc.communicate()[0]
    # return to where i was
    # os.chdir(prev_path)
    # get rid of temporary file
    temp.close()
    # return the parse trees
    return chunks([l for l in output.splitlines() if l[:3] == '(S1'],n)


