#!/usr/bin/python
import re
import IPython
import sys
sys.path.insert(1,'..')
from automain import automain

POS_rules = [
    (' ',None),
    ('the','DET'),
    ('object','NOUN'),
    ('object','VERB'),
    ('s','(PLURAL)'),
    ('behind','RELATION'),
]

# constuction_rules

def all_possible_POS(text_string):
    print 'text_string',text_string
    if text_string.strip() == '':
        return [[]]

    sequences = []
    for left_side,right_side in POS_rules:
        print 'sequences',sequences
        match = re.match(left_side, text_string)
        if match:
            new_sequences = all_possible_POS(text_string[match.end():])
            print 'new_sequences',new_sequences
            for sequence in new_sequences:
                sequence.insert(0,right_side)
            print 'new_sequences',new_sequences
            sequences.extend(new_sequences)
    new_sequences = []
    for sequence in sequences:
        new_sequences.append(filter(None, sequence))
    return new_sequences

@automain
def main():
    IPython.embed()