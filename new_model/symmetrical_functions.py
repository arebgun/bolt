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

    def decay_envelope(x, loc, scale):
        scale /= math.e
        return np.exp(-np.abs(x-loc)/(2*scale))

    def logistic_bell(x, loc, scale):
        return np.exp(-np.pi*(x-loc)/(scale*sqrt3))/(((1+np.exp(-np.pi*(x-loc)/(scale*sqrt3)))/2.)**2)

    def gaussian_bell(x, loc, scale):
        return np.exp(-(x-loc)**2/(2*scale**2))

    def sech_bell(x, loc, scale):
        return 1/(np.cosh((np.pi*(x-loc)/(2*scale*sqrt3)))**2)

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

    def bell_initial_params(x,y):
        return sample_mean_and_std(x[np.where(y==1)])


    functions = {
        'gaussian':  {'func':gaussian_bell,
                      'bounds':[(None,None),(0,None)],
                      'initial':bell_initial_params},
        'logistic':  {'func':logistic_bell, 
                      'bounds':[(None,None),(0,None)],
                      'initial':bell_initial_params},
        'exp decay': {'func':decay_envelope, 
                      'bounds':[(None,None),(0,None)],
                      'initial':bell_initial_params},
        'sech':      {'func':sech_bell,
                      'bounds':[(None,None),(0,None)],
                      'initial':bell_initial_params},
        # 'von mises': {'func':von_mises_function, 
        #               'bounds':[(None,None),(0,None)],
        #               'initial':angular_bell_initial_params},
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
    # p0 = [1,1]
    print 'p',loc0,scale0
    loc1, scale1 = initial_params(train_xs,train_ys)
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