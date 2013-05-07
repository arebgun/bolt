#!/usr/bin/python
import shelve
from matplotlib import pyplot as plt
import argparse
import numpy as np
import sys
sys.path.append("..")
from semantics import relation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', type=str, nargs='+')
    parser.add_argument('-w', '--window-size', type=int, default=200)
    parser.add_argument('--post', action='store_true')
    args = parser.parse_args()

    object_answers       = []
    object_distributions = []
    object_rel_types     = []

    test_object_answers       = []
    test_object_distributions = []
    test_object_rel_types     = []

    for filename in args.filenames:
        f = shelve.open(filename)

        if f.has_key('object_answers'):
            object_answers       += f['object_answers']
            object_distributions += f['object_distributions']

            if f.has_key('object_rel_types'):
                object_rel_types     += f['object_rel_types']
            else:
                object_rel_types     += [None]*len(f['object_answers'])

            if f.has_key('object_sentences'):
                object_sentences = f['object_sentences']
                print len(set(object_sentences)), 'unique object descriptions'

        if f.has_key('test_object_answers'):
            test_object_answers       += f['test_object_answers']
            test_object_distributions += f['test_object_distributions']

            if f.has_key('test_object_sentences'):
                test_object_sentences = f['test_object_sentences']
                print len(set(test_object_sentences)), 'unique object descriptions'

    window = args.window_size
    def running_avg(arr):
        return  [None] + [np.mean(arr[:i]) for i in range(1,window)] + [np.mean(arr[i-window:i]) for i in range(window,len(arr))]
        # return  [None] + [np.median(arr[:i]) for i in range(1,window)] + [np.median(arr[i-window:i]) for i in range(window,len(arr))]

    if f.has_key('object_answers'):
        correct               = [(answer == distribution[0][1]) for answer, distribution in zip(object_answers,object_distributions)]
        avg_correct           = running_avg(correct)

    if f.has_key('test_object_answers'):
        test_correct               = [(answer == [x[1] for x in distribution[0]]) for answer, distribution in zip(test_object_answers,test_object_distributions)]
        test_avg_correct           = np.mean(test_correct)

    # initial_training = f['initial_training']
    # cheating = f['cheating']
    # explicit_pointing = f['explicit_pointing']
    # ambiguous_pointing = f['ambiguous_pointing']

    title = ''
    # if initial_training:
    #     title = 'Initial Training'
    # if cheating:
    #     title = 'Cheating (Telepathy)'
    # if explicit_pointing:
    #     title = 'Explicit Pointing\n(Telepath Landmark only)'
    # if ambiguous_pointing:
    #     title = 'Ambiguous Pointing'
    plt.ion()

    if f.has_key('object_answers'):
        n=len(correct)
        plt.figure()
        plt.title(title)
        # plt.subplot(211)
        # plt.plot(correct[:n], 'o-', color='RoyalBlue')
        plt.plot(avg_correct[:n], 'x-', color='Orange')
        plt.plot(test_avg_correct, 'o-', color='RoyalBlue')
        plt.ylabel('Correct')
        plt.ylim([0,1])
        # plt.subplot(212)
        # plt.plot(distances[:n], 'o-', color='RoyalBlue')
        # plt.plot(avg_distances[:n], 'x-', color='Orange')
        # plt.ylabel('Distance')
        # plt.figure()
        # plt.title('Number of relations (object task)')
        # plt.plot(object_num_tofrom, 'x-', color='Blue', label='To/From')
        # plt.plot(object_num_lrfb, 'x-', color='Green', label='L/R/F/B')
        # plt.plot(object_num_on, 'x-', color='Salmon', label='On')
        # leg = plt.legend()
        # leg.draggable(state=True)


    plt.ioff()
    plt.show()
    plt.draw()

    f.close()

