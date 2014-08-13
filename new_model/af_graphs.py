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

import functools as ft

def angular_mean(x):
    return np.arctan2(np.sin(x).sum()/len(x), np.cos(x).sum()/len(x))

@automain
def main():

    def angle_diff(angles1, angle2):
        return np.abs((angles1-angle2+np.pi)%(2*np.pi)-np.pi)

    def angle_diff2(angles1, angle2):
        return np.abs((angles1-angle2+180)%(360)-180)

    def decay_envelope(x, loc, scale):
        return np.exp(-angle_diff2(x,loc)/(2*scale))

    def von_mises_function(x, loc, scale):
        return np.exp(scale*(np.cos(np.radians(x)-np.radians(loc))-1))

    def logistic_function(x, loc, scale):
        return 1./(1+np.exp(-0.01*scale*(x-loc)))

    def sech_family_distribution(x, loc, scale, power=1):
        return 1/(np.cosh((angle_diff2(x,loc)/scale))**power)

    def double_logistic_function(x, loc, scale):
        return logistic_function(x,loc-2*scale,10*scale)*\
               logistic_function(x,loc+2*scale,-10*scale)

    uared = np.array((204,0,51))/255.

    function = logistic_function

    fig0, ax0 = plt.subplots()
    names = ['black', 'red', 'green', 'blue', 'white']
    heights = [0,1,0,0,0]
    ind = np.arange(len(names))
    ax0.bar(ind, heights, width=1.0, color=uared, edgecolor='none')
    ax0.set_ylim((0,1.1))
    ax0.set_xticks(ind+0.5)
    ax0.set_xticklabels( names )
    ax0.set_xlim((0,5))

    fig1, ax1 = plt.subplots()
    x = np.arange(-180, 180, 0.1)
    loc0 = 100
    scale0 = 7
    f = function(x, loc=loc0, scale=scale0)
    l1, = ax1.plot(x, f, lw=2, color=uared)
    ax1.fill_between(x, 0, f, color=uared)
    ax1.axis([-180, 180, 0, 1])


    function = sech_family_distribution

    fig2, ax2 = plt.subplots()
    x = np.arange(-180, 180, 0.1)
    loc0 = 100
    scale0 = 30
    f = function(x, loc=loc0, scale=scale0)
    l2, = ax2.plot(x, f, lw=2, color=uared)
    ax2.fill_between(x, 0, f, color=uared)
    ax2.axis([-180, 180, 0, 1])


    plt.show()