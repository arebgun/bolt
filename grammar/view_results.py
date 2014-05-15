#!/usr/bin/python

import shelve
import numpy as np
from matplotlib import pyplot as plt
import argparse
import IPython
from collections import defaultdict

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

def separate_answers(filename):
    f = shelve.open(filename)
    all_answers = f['all_answers']
    f.close()

    a = sorted([val for tup in zip(*all_answers) for val in tup])
    d = defaultdict(list)
    for time,clas,string,sempole,answer in a:
        d[string].append(answer)
    for key, value in d.items():
        d[key] = np.array(value,dtype=float)
        d[key][np.where(np.isnan(d[key]))] = 0.0
    return d


def something():
    nouns= ['noun1.shelf','noun2.shelf','noun3.shelf','noun4.shelf','noun5.shelf']
    adjs = ['adj1.shelf','adj2.shelf','adj3.shelf','adj4.shelf','adj5.shelf']
    adjsandnouns = ['adjsandnouns1.shelf','adjsandnouns2.shelf','adjsandnouns3.shelf','adjsandnouns4.shelf','adjsandnouns5.shelf']
    rels = ['rel1.shelf','rel2.shelf','rel3.shelf','rel4.shelf','rel5.shelf']
    rels2 = ['learning_game_results_best6.shelf']

    for names, maxx, color, label, avgwin in [
                                      (nouns,1.0,(.9,.6,.2),1.4, 100),
                                      (adjs,0.95, (.9,.5,.2),1.5, 100),
                                      (adjsandnouns, None, (.9,.2,.2),1.6, 100),
                                      # (rels,0.9,(.35,.8,.9),3.2, 100),
                                      # (rels2, None, (.35,.6,.9), 4.2, 200)
                                      ]:

        answerss = [get_answers(f) for f in names]
        sums = sum(answerss)
        avgs = sums/len(answerss)
        avgs = running_avg(avgs,avgwin)
        plt.plot(range(len(avgs)),avgs,color=color,linewidth=4,label='Experiment %s'%label)
        # plt.scatter([315],[maxx],s=60,marker='<',c=color,edgecolor='none')
        plt.ylabel('Fraction correct')
        plt.xlabel('Number of utterances observed')
        plt.xlim([0,len(avgs)])
        plt.ylim([0,1])
        plt.legend(loc='lower right')
    plt.show()

def separate_lines():
    rels = ['rel1.shelf','rel2.shelf','rel3.shelf','rel4.shelf','rel5.shelf']
    ds = [separate_answers(f) for f in rels]
    strings = ['near to', 'far from', 'to the left of', 'to the right of', 'to the front of', 'to the back of']
    d2 = {}
    for string in strings:
        l = max([len(d[string]) for d in ds])
        arr = np.ma.empty((5,l))
        arr.mask = True
        for i,d in enumerate(ds):
            arr[i,:d[string].shape[0]] = d[string]
        d2[string] = arr.mean(axis=0)
        print string
        print d2[string]
        avgs = running_avg(d2[string],30)
        plt.plot(range(len(avgs)),avgs, label=string)
        # plt.scatter([315],[maxx],s=60,marker='<',c=color,edgecolor='none')
        plt.ylabel('Fraction correct')
        plt.xlabel('Number of utterances observed')
        # plt.xlim([0,len(avgs)])
    plt.ylim([0,1])
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    # separate_lines()
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', type=str, nargs='+')
    args = parser.parse_args()


    answerss = [get_answers(f) for f in args.filenames]
    for answers in answerss:
        print np.mean(answers)
    sums = sum(answerss)
    print len(sums)
    avgs = sums/len(answerss)
    avgs = running_avg(avgs,150)
    plt.plot(range(len(avgs)),avgs)
    # plt.scatter([315],[maxx],s=60,marker='<',c=color,edgecolor='none')
    plt.ylabel('Fraction correct')
    plt.xlabel('Number of utterances observed')
    plt.xlim([0,len(avgs)])
    plt.ylim([0,1])
    # plt.legend(loc='lower right')
    plt.show()

