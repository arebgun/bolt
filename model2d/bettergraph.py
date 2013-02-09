#!/usr/bin/python
import shelve
from matplotlib import pyplot as plt
import argparse
import numpy as np
import sys
sys.path.append("..")
from semantics import relation
from collections import Counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', type=str, nargs='+')
    parser.add_argument('-w', '--window-size', type=int, default=200)
    parser.add_argument('--post', action='store_true')
    args = parser.parse_args()

    object_answers       = []
    object_distributions = []
    object_ids           = []
    object_sentences     = []

    test_object_answers       = []
    test_object_distributions = []
    test_object_ids           = []
    test_object_sentences     = []

    chunks = []

    for filename in args.filenames:
        f = shelve.open(filename)

        turk_answers = f['turk_answers']

        object_answers       += f['object_answers']
        object_distributions += f['object_distributions']
        object_ids           += f['object_ids']
        object_sentences     += f['object_sentences']

        test_object_answers       += f['test_object_answers']
        test_object_distributions += f['test_object_distributions']
        test_object_ids           += f['test_object_ids']
        test_object_sentences     += f['test_object_sentences']

        chunks += f['chunks']
    # chunks = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,95,75,55,35]
    xs = np.cumsum(chunks)

        # chunksize = 5*24
        # numchunks = int(3015/chunksize)
        # chunks    = [chunksize]*numchunks + [3015-chunksize*numchunks]

    for key in turk_answers:
        total = float(len(turk_answers[key]))
        turk_answers[key] = Counter(turk_answers[key])
        for answer in turk_answers[key]:
            turk_answers[key][answer] = turk_answers[key][answer]/total
        # print turk_answers[key]
        # print turk_answers[key][u'object_1']

    # exit()
    window = args.window_size
    def running_avg(arr,window):
        return  [np.mean(arr[:i]) for i in range(1,window)] + [np.mean(arr[i-window:i]) for i in range(window,len(arr))]


    top_answer = []
    if f.has_key('object_answers'):
        correct               = [(answer[1] == distribution[0][2]) for answer, distribution in zip(object_answers,object_distributions)]
        avg_correct           = running_avg(correct,window)

        test_avgs = []
        filtered_test_avgs = []
        turk_avgs = []
        for answers,distributions,ids in zip(test_object_answers,test_object_distributions,test_object_ids):
            # for answer, distribution in zip(answers,distributions):
            #     print answer, distribution[0]
            test_avgs.append(  np.mean( [(answer[0] == distribution[0][1]) for answer, distribution in zip(answers,distributions) if answer] )  )
            # filtered_test_avgs.append(  np.mean( [(answer[1] == distribution[0][2]) for answer, distribution in zip(answers,distributions) if answer[1]])  )

            turk_correct = []
            for answer,distribution,eyedee in zip(answers,distributions,ids):
                if answer:
                    top_answer.append( turk_answers[eyedee][answer[0]] )
                    if turk_answers[eyedee].most_common(1)[0][1] >= 0:
                        turk_correct.append( turk_answers[eyedee].most_common(1)[0][0] == distribution[0][1] )
                else:
                    turk_correct.append(False)
            turk_avgs.append( np.mean(turk_correct) )
    # print np.mean(top_answer)
    title = ''
    plt.ion()

    n=len(correct)
    plt.figure()
    plt.title(title)
    plt.plot(avg_correct[:n], '-', linewidth=4, color='DimGray', label='Training (rolling mean)')
    # print test_avgs
    # print turk_avgs
    plt.plot(xs[:len(test_avgs)],test_avgs, 'o', markersize=10, color='RoyalBlue', label='Test set ("right" answers)')
    # plt.plot(xs[:len(test_avgs)],filtered_test_avgs, 'o', color='RoyalBlue', label='Test set')
    plt.plot(xs[:len(turk_avgs)],turk_avgs, 'o', markersize=10, color='Orange', label='Test set (Turk answers)')
    plt.axhline(np.mean(top_answer), linewidth=2, color='Red', label="Turker accuracy")
    leg = plt.legend()
    leg.draggable(state=True)
    plt.ylabel('Percent correct')
    plt.xlabel('Training instances')
    plt.ylim([0,1])
    plt.grid(True)

    plt.ioff()
    plt.show()
    plt.draw()

    f.close()

