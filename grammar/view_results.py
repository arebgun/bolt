#!/usr/bin/python

import shelve
import numpy as np
from matplotlib import pyplot as plt
import argparse
import IPython

def running_avg(arr,window):
    return  [np.mean(arr[:i]) for i in range(1,window)] + [np.mean(arr[i-window:i]) for i in range(window,len(arr))]

def get_answers(filename):
    f = shelve.open(filename)
    all_answers = f['all_answers']
    f.close()

    a = [val for tup in zip(*all_answers) for val in tup]
    times,classes,strings,sempoles,answers = zip(*sorted(a))
    answers = np.array(answers,dtype=float)
    answers[np.where(np.isnan(answers))] = 0.0
    return answers

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('filenames', type=str, nargs='+')
    # args = parser.parse_args()
    nouns= ['noun1.shelf','noun2.shelf','noun3.shelf','noun4.shelf','noun5.shelf']
    adjs = ['adj1.shelf','adj2.shelf','adj3.shelf','adj4.shelf','adj5.shelf']
    rels = ['rel1.shelf','rel2.shelf','rel3.shelf','rel4.shelf','rel5.shelf']

    for names, maxx, color, label in [(nouns,1.0,(.35,.7,.9),1),(adjs,0.95, (.9,.5,.2),2),(rels,0.9,(0,0,0),3)]:

        answerss = [get_answers(f) for f in names]
        sums = sum(answerss)
        avgs = sums/len(answerss)
        avgs = running_avg(avgs,50)
        plt.plot(range(len(avgs)),avgs,color=color,linewidth=3,label='Experiment %s'%label)
        # plt.scatter([315],[maxx],s=60,marker='<',c=color,edgecolor='none')
        plt.ylabel('Fraction correct')
        plt.xlabel('Number of utterances observed')
        plt.xlim([0,len(avgs)])
        plt.ylim([0,1])
        plt.legend(loc='lower right')
    plt.show()




