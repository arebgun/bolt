#!/usr/bin/python

#!/usr/bin/python
from __future__ import division
import sys
sys.path.insert(1,"..")

from automain import automain

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wdg
# from matplotlib.widgets import Slider, Button, RadioButtons

import scipy.stats as st
import scipy.constants as con

import scipy.optimize as opt

import random

import math

import IPython


def logistic_pdf(x, loc, scale):
    return np.exp(-(x-loc)/scale)/(scale*(1+np.exp(-(x-loc)/scale))**2)

def logistic_cdf(x, loc, scale):
    return 1./(1+np.exp(-(x-loc)/scale))

@automain
def main():


    def generate_training_data(x, function, params, num_trials, xnoise=0):
        xs, ys = \
            zip(*[(_x,np.random.random() <= function(_x+st.norm.rvs(0,xnoise), 
                                                     *params)) 
               for _x in x
               for i in range(num_trials)#range(random.choice(range(num_trials)))
               ])
        xs = np.array(xs)
        ys = np.array(ys)
        return xs, ys

    # plt.title('Progression of Tall')
    fig, (ax1, ax2) = plt.subplots(2,1)
    # plt.subplots_adjust(left=0.25, bottom=0.25)

    min_feet = 4
    max_feet = 8
    inches_in_foot = 12
    inches = 1

    ax1.axis([min_feet*inches_in_foot, max_feet*inches_in_foot, 0, 0.5])
    ax1.set_xlabel('Male Height (in)')
    ax1.set_ylabel('Probability Density')
    ax2.axis([min_feet*inches_in_foot, max_feet*inches_in_foot, 0, 1])
    ax2.set_xlabel('Male Height (in)')
    ax2.set_ylabel('Cumulative Density')
    
    mean_male_height = 69.4 * inches
    sterr_male_height = 0.07 * inches
    sample_size = 4482
    stdev_male_height = sterr_male_height*np.sqrt(sample_size)

    mean_heights = [mean_male_height]
    stdev_heights = [stdev_male_height]
    x = np.arange(min_feet*inches_in_foot, max_feet*inches_in_foot, 0.1)

    names = ['normal', 'tall', 'x2 tall', 'x3 tall', 'x4 tall',
             'x5 tall', 'x6 tall', 'x7 tall', 'x8 tall', 'x9 tall']

    max_times = 20
    while len(mean_heights) < max_times:
        mean_height = mean_heights[-1]
        stdev_height = stdev_heights[-1]

        # print 'height mean:', np.floor(mean_height/12.), mean_height%12
        print mean_height
        # if len(mean_heights)>1:
        #     mean_diff = mean_heights[-1]-mean_heights[-2]
        #     print 'diff:', mean_diff, mean_diff/stdev_height
        # else:
        #     print
        # print 'height std: ', np.floor(stdev_height/12.), stdev_height%12

        height_dist = st.norm(loc=mean_height, scale=stdev_height)
        f1 = height_dist.pdf(x)
        f2 = height_dist.cdf(x)
        l1, = ax1.plot(x, f1, lw=2)
        l2, = ax2.plot(x, f2, lw=2)

        # f3 = logistic_pdf(x, loc=mean_height, scale=stdev_height/con.golden)
        # f4 = logistic_cdf(x, loc=mean_height, scale=stdev_height/con.golden)
        # ax1.plot(x, f3, lw=2)
        # ax2.plot(x, f4, lw=2)

        sample_heights = height_dist.rvs(size=sample_size)
        sample_taller_probability = height_dist.cdf(sample_heights)
        sample_taller = np.array([random.random() < tp for tp in sample_taller_probability])
        # sample_taller = np.array([h > mean_height for h in sample_heights])
        taller_heights = sample_heights[np.where(sample_taller)]

        # if len(mean_heights) == 2:
        #     ax1.hist(taller_heights, bins=15, normed=True)

        taller_mean_height = taller_heights.mean()
        taller_stdev_height = taller_heights.std()

        mean_heights.append(taller_mean_height)
        stdev_heights.append(taller_stdev_height)

    ax1.legend(names)
    # plt.show()

    IPython.embed()