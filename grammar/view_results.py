#!/usr/bin/python

import sys
sys.path.insert(1,'..')
import semantics as sem

import automain
import utils
# import random
import common as cmn
import language_user as lu
import lexical_items as li
import constructions as st
import construction as cs
import constraint

import shelve
import numpy as np
from matplotlib import pyplot as plt
import argparse
import IPython
from collections import defaultdict

def running_avg(arr,window,repeat=1):
    for _ in range(repeat):
        arr = [np.mean(arr[i/2:i]) for i in range(1,min(window*2,len(arr)))] + [np.mean(arr[i-window:i]) for i in range(window*2,len(arr))]
    return  arr
    # return [np.mean(arr) for _ in arr]

def middle_avg(arr,window):
    return [np.mean(arr[max(0,i-window/2):min(len(arr),i+window/2)]) for i in range(len(arr))]

def get_answers(filename):
    f = shelve.open(filename)
    all_answers = f['all_answers']
    f.close()

    a = [val for tup in zip(*all_answers) for val in tup]
    times,classes,strings,sempoles,answers,baselines = zip(*sorted(a))
    answers = np.array(answers,dtype=float)
    answers[np.where(np.isnan(answers))] = 0.0
    baselines = zip(*baselines)
    for i in range(len(baselines)):
        baselines[i] = np.array(baselines[i], dtype=float)
        baselines[i][np.where(np.isnan(baselines[i]))] = 0.0
    return answers, baselines

def separate_answers(filename):
    f = shelve.open(filename)
    all_answers = f['all_answers']
    f.close()

    a = sorted([val for tup in zip(*all_answers) for val in tup])
    d = defaultdict(list)
    for time,clas,string,sempole,answer,baselines in a:
        d[string.replace(' ','') if string else None].append((answer,baselines[0]))
    for string in d:
        d[string] = zip(*d[string])
    for key, (value1,value2) in d.items():
        d[key] = (np.array(value1,dtype=float),np.array(value2,dtype=float))
        d[key][0][np.where(np.isnan(d[key][0]))] = 0.0
        d[key][1][np.where(np.isnan(d[key][1]))] = 0.0
    return d, len(a)


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

import matplotlib.cm as cm
import itertools
def plot_by_type(args):
    assert(len(args.filenames) == 1)
    relmap = {'a t':'at','to w a rds':'towards','a bove':'above'}
    # for f in args.filenames:
    f = args.filenames[0]
    separated, total = separate_answers(f)
    # fig = plt.figure()
    keys, _ = zip(*sorted([s for s in separated.items() if s[0] and len(s[0])<20],key=lambda x: len(x[1][0]),reverse=True))
    markers = itertools.cycle(['o','s','v'])
    def plot_things(x):
        for rel, i in zip(keys, np.linspace(0,256,len(keys)).astype(int)):
            # if rel and len(rel) < 20 and len(separated[rel]) > 35:
            avgs = running_avg(separated[rel][x],args.window,repeat=3)
            # indices = np.linspace(0,total,len(avgs)).astype(int)
            indices = range(len(avgs))
            plt.plot(indices,avgs,c=cm.gist_rainbow(i),linewidth=3)#,label=(relmap[rel] if rel in relmap else rel))
            plt.plot(indices[-1:],avgs[-1:],c=cm.gist_rainbow(i),marker=markers.next(),markersize=10,label=(relmap[rel] if rel in relmap else rel))

        plt.subplots_adjust(right=0.8)
        plt.ylabel('Fraction correct')
        plt.xlabel('Number of utterances observed')
        # plt.xlim([0,len(avgs[:n])])
        plt.ylim([0,1])
        leg = plt.legend(loc='lower right', fancybox=True)
        leg.draggable(True)
        plt.title(args.title + ' - by phrase - ' + ('Reagent' if x==0 else 'PMI'))
        plt.show()

    g = shelve.open(f)
    all_answers = g['all_answers']
    g.close()

    a = sorted([val for tup in zip(*all_answers) for val in tup],reverse=True)

    # for tup in a:
    #     print tup[2]
    #     x = raw_input()
    #     if x == 'x':
    #         exit()

    for key in keys:
        print key
        count = 0
        for tup in a:
            if (tup[2] is not None) and tup[2].replace(' ','') == key:
                print tup[3]
                count += 1
                if count >= 5:
                    break
        print
    # IPython.embed()
    # exit()
    plot_things(0)
    plot_things(1)


if __name__ == '__main__':
    # separate_lines()
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--title', type=str, default='')
    parser.add_argument('-w', '--window', type=int, default=100)
    parser.add_argument('-n', '--number', type=int, default=None)
    parser.add_argument('filenames', type=str, nargs='+')
    args = parser.parse_args()

    # plot_by_type(args)

    answerss = []
    baseliness = []
    for f in args.filenames:
        answers, baselines = get_answers(f)
        answerss.append(answers)
        baseliness.append(baselines)
    baseliness = zip(*baseliness)
    # answerss, baseline1s = zip(*[get_answers(f) for f in args.filenames])
    for answers in answerss:
        print np.mean(answers)
    for baselines in baseliness:
        print np.mean(baselines)
    sums = sum(answerss)
    avgs = sums/len(answerss)
    print len(avgs)
    bavgss = []
    for baselines in baseliness:
        bavgss.append(running_avg(sum(baselines)/len(baselines), args.window,repeat=3))
    # b1sums = sum(baseline1s)
    # print len(sums), len(b1sums)
    # b1avgs = b1sums/len(baseline1s)
    avgs = running_avg(avgs,args.window,repeat=3)
    if args.number is None:
        n = len(avgs)
    else:
        n = args.number
    # b1avgs = running_avg(b1avgs,args.window)
    fig = plt.figure()
    lines = plt.plot(range(n),avgs[:n],color=(.35,.6,.9),linewidth=3,label='reagent')
    # for i, bavgs in enumerate(bavgss):
    lines.extend( plt.plot(range(n),bavgss[0][:n],color=(.9,.6,.2),linewidth=3,label='pmi/bayes') )
    # lines.extend( plt.plot([0,n-1],[0.651032,0.651032],linewidth=3,label='turkers') )
    # plt.scatter([315],[maxx],s=60,marker='<',c=color,edgecolor='none')
    plt.ylabel('Fraction correct')
    plt.xlabel('Number of utterances observed')
    plt.xlim([0,len(avgs[:n])])
    plt.ylim([0,1])
    leg = plt.legend(loc='lower right', fancybox=True)
    leg.draggable(True)
    # leg.pickable()


    # we will set up a dict mapping legend line to orig line, and enable
    # picking on the legend line
    # lines = [line1, line2]
    lined = dict()
    for legline, origline in zip(leg.get_lines(), lines):
        legline.set_picker(5)  # 5 pts tolerance
        lined[legline] = origline


    def onpick(event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        if event.artist is leg:
            return
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.title(args.title)

    # i=10
    # a = plt.axes([0.4, 0.3, .4, .4])#, axisbg='y')
    # plt.plot(range(i),avgs[:i])
    # plt.plot(range(i),bavgss[0][:i])
    # # title('Impulse response')
    # plt.title('First 10 iterations')
    # plt.setp(a)

    plt.show()

