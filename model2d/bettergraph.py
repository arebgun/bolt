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
    object_sentences     = []

    test_object_answers       = []
    test_object_distributions = []
    test_object_sentences     = []

    chunks = []

    for filename in args.filenames:
        f = shelve.open(filename)

        object_answers       += f['object_answers']
        object_distributions += f['object_distributions']
        object_sentences     += f['object_sentences']

        test_object_answers       += f['test_object_answers']
        test_object_distributions += f['test_object_distributions']
        test_object_sentences     += f['test_object_sentences']

        chunks += f['chunks']
    chunks = [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,95,75,55,35]
    xs = np.cumsum(chunks)

        # chunksize = 5*24
        # numchunks = int(3015/chunksize)
        # chunks    = [chunksize]*numchunks + [3015-chunksize*numchunks]

    window = args.window_size
    def running_avg(arr,window):
        return  [np.mean(arr[:i]) for i in range(1,window)] + [np.mean(arr[i-window:i]) for i in range(window,len(arr))]


    if f.has_key('object_answers'):
        correct               = [(answer == distribution[0][1]) for answer, distribution in zip(object_answers,object_distributions)]
        avg_correct           = running_avg(correct,window)

        test_avgs = []
	filtered_test_avgs = []
        for answers,distributions in zip(test_object_answers,test_object_distributions):
            test_avgs.append(  np.mean( [(answer == distribution[0][1]) for answer, distribution in zip(answers,distributions)] )  )
	    filtered_test_avgs.append(  np.mean( [(answer == distribution[0][1]) for answer, distribution in zip(answers,distributions) if answer])  )

        for answers in zip(*test_object_answers):
            print answers

        # distances = []
        # for answer, distribution in zip(object_answers,object_distributions):
        #     lmkprob, lmknum = zip(*distribution)
        #     lmkprob = np.array(lmkprob)/sum(lmkprob)
        #     distances.append( lmkprob[0] - lmkprob[lmknum.index(answer)] )
        #     # print answer, zip(lmknum,lmkprob)
        # avg_distances         = running_avgs(distances)
        # avg_distances = []
    
    title = ''
    plt.ion()

    n=len(correct)
    plt.figure()
    plt.title(title)
    plt.plot(avg_correct[:n], 'x-', color='Orange', label='Training (rolling mean)')
    print test_avgs
#    plt.plot(xs[:len(test_avgs)],test_avgs, 'o', color='RoyalBlue')
    plt.plot(xs[:len(test_avgs)],filtered_test_avgs, 'o', color='RoyalBlue', label='Test set')
    leg = plt.legend()
    leg.draggable(state=True)
    plt.ylabel('Percent correct')
    plt.ylim([0,1])
    plt.grid(True)

    plt.ioff()
    plt.show()
    plt.draw()

    f.close()

