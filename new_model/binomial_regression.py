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

    def binom_errfunc(params, func, xdata, ydata):
        epsilon = 0.000000001
        probs = func(xdata, *params)
        probs[np.where(probs==0)] += epsilon
        probs[np.where(probs==1)] -= epsilon
        # return np.product((probs**ydata)*((1-probs)**(1-ydata)))
        return -(ydata*np.log(probs) + (1-ydata)*np.log(1-probs)).sum()

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

    def fit_function(opt_function,p0,function,train_xs,train_ys,**kwargs):
        print 'p0',p0
        p1 = opt_function(func=binom_errfunc,
                          x0=p0,
                          args=(function,train_xs,train_ys),
                          **kwargs)[0]
        print 'p1',p1
        return p1

    def bell_initial_params(x,y):
        return x[np.where(y==1)].mean(), 1

    def angular_bell_initial_params(x,y):
        return np.degrees(angular_mean(np.radians(x[np.where(y==1)]))), 1

    def sigmoid_initial_params(x,y):
        sort_i = np.argsort(x)
        sorted_x = x[sort_i]
        sorted_y = y[sort_i]
        diffs = np.append(sorted_y[1:]-sorted_y[:-1], False)
        diff_x = sorted_x[np.where(diffs)]
        initial_loc = diff_x.mean()
        initial_scale = 185./diff_x.std()
        mean_diff = sorted_x[np.where(sorted_y)].mean() - \
                    sorted_x[np.where(np.logical_not(sorted_y))].mean()
        initial_scale = math.copysign(initial_scale,mean_diff)
        return initial_loc, initial_scale


    functions = {
        'logistic':  {'func':logistic_function, 
                      'bounds':[(None,None),(None,None)],
                      'initial':sigmoid_initial_params},
        'sech1':     {'func':sech_family_distribution,
                      'bounds':[(None,None),(None,None)],
                      'initial':bell_initial_params},
        'sech2':     {'func':ft.partial(sech_family_distribution,power=2),
                      'bounds':[(None,None),(None,None)],
                      'initial':bell_initial_params},
        'sech/2':     {'func':ft.partial(sech_family_distribution,power=0.5),
                      'bounds':[(None,None),(None,None)],
                      'initial':bell_initial_params},
        'double log':{'func':double_logistic_function,
                      'bounds':[(None,None),(None,None)],
                      'initial':bell_initial_params},
        'probit':    {'func':st.norm.cdf, 
                      'bounds':[(None,None),(0,None)],
                      'initial':sigmoid_initial_params},
        'von mises': {'func':von_mises_function, 
                      'bounds':[(None,None),(0,None)],
                      'initial':angular_bell_initial_params},
        'exp decay': {'func':decay_envelope, 
                      'bounds':[(None,None),(0,None)],
                      'initial':angular_bell_initial_params},
    }

    global function
    global x
    global bounds
    global initial_params
    function = functions['logistic']['func']
    bounds = functions['logistic']['bounds']

    fig0, ax0 = plt.subplots()


    opt_function = opt.fmin_l_bfgs_b
    fig, (ax1, ax2) = plt.subplots(2,1)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    x = np.arange(-180, 180, 0.1)
    loc0 = 0
    scale0 = 10
    f = function(x, loc=loc0, scale=scale0)
    l1, = ax1.plot(x, f, lw=2, color='red')
    ax1.axis([-180, 180, 0, 1])

    initial_params = functions['logistic']['initial']

    xnoise0 = 0
    num_trials = 3
    train_xs, train_ys = generate_training_data(x, function, 
                                                [loc0,scale0],
                                                num_trials,
                                                xnoise=xnoise0)
    # p0 = [1,1]
    print 'p',loc0,scale0
    p0 = initial_params(train_xs,train_ys)
    kwargs = dict(approx_grad=True, disp=True, bounds=bounds, maxfun=5)
    p1 = fit_function(opt_function,p0,function,train_xs,train_ys,**kwargs)
    f2 = function(x,*p1)
    l2, = ax2.plot(x, f2, lw=2, color='red')
    ax2.axis([-180, 180, 0, 1])

    l0, = ax0.plot(x, f2, lw=2, color='red')
    ax0.set_ylim((-0.1,1.1))
    # l01 = ax0.scatter(train_xs[100:],train_ys[100:])
    # ax0.axis([-0.2, 1.4, -0.1, 1.1])




    axcolor = 'lightgoldenrodyellow'
    axloc = plt.axes([0.25, 0.07, 0.65, 0.03], axisbg=axcolor)
    axscale  = plt.axes([0.25, 0.12, 0.65, 0.03], axisbg=axcolor)
    axnoise = plt.axes([0.25, 0.17, 0.65, 0.03], axisbg=axcolor)

    sloc = wdg.Slider(axloc, 'Loc', -180, 180, valinit=loc0)
    sscale = wdg.Slider(axscale, 'Scale', -20, 20, valinit=scale0)
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
        p1 = initial_params(train_xs, train_ys)
        # p1 = fit_function(opt_function,p0,function,train_xs,train_ys,**kwargs)
        f2 = function(x,*p1)
        l2.set_ydata(f2)
        l0.set_ydata(f2)
        # amp = samp.val
        # freq = sfreq.val
        # l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        fig.canvas.draw_idle()
        fig0.canvas.draw_idle()
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