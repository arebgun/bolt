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

def sample_mean_and_std(x):
    N = x.shape[0]
    mu = x.mean()
    std = np.sqrt(((x-mu)**2).sum()/(N-1))
    return mu, std
    

@automain
def main():

    sqrt3 = np.sqrt(3)

    def logistic_sigmoid(x, loc, scale):
        return 1./(1+np.exp(-(x-loc)/scale))


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

    def sigmoid_initial_params(x,y):
        sort_i = np.argsort(x)
        sorted_x = x[sort_i]
        sorted_y = y[sort_i]
        diffs = sorted_y[1:]-sorted_y[:-1]
        # diff_x = sorted_x[np.where(diffs)]
        diff_x = (sorted_x[np.where(diffs)]+
                 sorted_x[np.where(diffs[1:])])/2.0
        initial_loc = diff_x.mean()
        initial_scale = diff_x.std()
        mean_diff = sorted_x[np.where(sorted_y)].mean() - \
                    sorted_x[np.where(np.logical_not(sorted_y))].mean()
        initial_scale = math.copysign(initial_scale,mean_diff)
        return initial_loc, initial_scale

    def logistic_sigmoid_initial_params(x,y):
        initial_loc, initial_scale = sigmoid_initial_params(x,y)
        return initial_loc, initial_scale*sqrt3/np.pi


    functions = {
        'logistic':  {'func':logistic_sigmoid, 
                      'bounds':[(None,None),(0,None)],
                      'initial':logistic_sigmoid_initial_params},
    }

    global function
    global x
    global bounds
    global initial_params
    function = functions['logistic']['func']
    bounds = functions['logistic']['bounds']

    fig, (ax1, ax2) = plt.subplots(2,1)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    x = np.arange(-180, 180, 0.1)
    loc0 = 0
    scale0 = 25
    f = function(x, loc=loc0, scale=scale0)
    l1, = ax1.plot(x, f, lw=3, color='red')
    ax1.axis([-180, 180, 0, 1])

    initial_params = functions['logistic']['initial']

    xnoise0 = 0
    num_trials = 1
    train_xs, train_ys = generate_training_data(x, function, 
                                                [loc0,scale0],
                                                num_trials,
                                                xnoise=xnoise0)

    inds = np.arange(len(train_xs))
    random.shuffle(inds)
    inds = inds[:100]
    train_xs, train_ys = train_xs[inds], train_ys[inds]

    # p0 = [1,1]
    print 'p0', loc0, scale0
    loc1, scale1 = initial_params(train_xs,train_ys)
    print 'p1', loc1, scale1
    print
    f2 = function(x, loc=loc1, scale=scale1)
    l2, = ax1.plot(x, f2, lw=2, color='blue')
    ax2.axis([-180, 180, 0, 1])


    axcolor = 'lightgoldenrodyellow'
    axloc = plt.axes([0.25, 0.07, 0.65, 0.03], axisbg=axcolor)
    axscale  = plt.axes([0.25, 0.12, 0.65, 0.03], axisbg=axcolor)
    axnoise = plt.axes([0.25, 0.17, 0.65, 0.03], axisbg=axcolor)

    sloc = wdg.Slider(axloc, 'Loc', -180, 180, valinit=loc0)
    sscale = wdg.Slider(axscale, 'Scale', -50, 50, valinit=scale0)
    snoise = wdg.Slider(axnoise, 'Noise', 0, 60, valinit=xnoise0)

    def update(val):
        loc = sloc.val
        scale = sscale.val
        xnoise = snoise.val
        f = function(x, loc=loc, scale=scale)
        l1.set_ydata(f)
        train_xs, train_ys = generate_training_data(x, function, 
                                                    [loc,scale],
                                                    num_trials,
                                                    xnoise=xnoise)
        print 'p',loc,scale
        loc1, scale1 = initial_params(train_xs, train_ys)
        print 'p1', loc1, scale1
        print
        f2 = function(x,loc=loc1,scale=scale1)
        l2.set_ydata(f2)
        # amp = samp.val
        # freq = sfreq.val
        # l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        fig.canvas.draw_idle()
    sloc.on_changed(update)
    sscale.on_changed(update)
    snoise.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = wdg.Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    def reset(event):
        sloc.reset()
        sscale.reset()
        snoise.reset()
    button.on_clicked(reset)

    rax = plt.axes([0.025, 0.5, 0.15, 0.2], axisbg=axcolor)
    funcradio = wdg.RadioButtons(rax, functions.keys(), 
                                 active=functions.keys().index('logistic'))
    def funcfunc(label):
        global function
        global bounds
        global initial_params
        function = functions[label]['func']
        bounds = functions[label]['bounds']
        initial_params = functions[label]['initial']
        update(None)
        fig.canvas.draw_idle()
    funcradio.on_clicked(funcfunc)

    # rax = plt.axes([0.025, 0.75, 0.15, 0.15], axisbg=axcolor)
    # squareradio = wdg.RadioButtons(rax, ('x', 'x^2'), active=0)
    # def squarefunc(label):
    #     update(None)
    #     fig.canvas.draw_idle()
    # squareradio.on_clicked(squarefunc)

    plt.show()