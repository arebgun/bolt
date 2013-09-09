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

@automain
def main():

    global function
    global x
    function = st.logistic
    fig, (ax1, ax2) = plt.subplots(2,1)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    _x = np.arange(-10, 10, 0.01)
    _x2 = _x**2
    x = _x
    loc0 = 0
    scale0 = 1
    f = function.pdf(x, loc=loc0, scale=scale0)
    Sf = function.cdf(x, loc=loc0, scale=scale0)
    # a0 = 5
    # f0 = 3
    # s = a0*np.sin(2*np.pi*f0*t)
    l1, = ax1.plot(x, f, lw=2, color='red')
    ax1.axis([-10, 10, 0, 1])
    l2, = ax2.plot(x, Sf, lw=2, color='red')
    ax2.axis([-10, 10, 0, 1])


    axcolor = 'lightgoldenrodyellow'
    axloc = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    axscale  = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

    sloc = wdg.Slider(axloc, 'Loc', -10, 10, valinit=loc0)
    sscale = wdg.Slider(axscale, 'Scale', 0, 10, valinit=scale0)

    def update(val):
        loc = sloc.val
        scale = sscale.val
        if function == st.norm:
            scale = scale*con.golden
        if function == st.cauchy:
            scale = scale*np.sqrt(con.golden)
        l1.set_ydata(function.pdf(x, df=100, loc=loc, scale=scale))
        l2.set_ydata(function.cdf(x, df=100, loc=loc, scale=scale))
        # amp = samp.val
        # freq = sfreq.val
        # l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        fig.canvas.draw_idle()
    sloc.on_changed(update)
    sscale.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = wdg.Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    def reset(event):
        sloc.reset()
        sscale.reset()
    button.on_clicked(reset)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
    colorradio = wdg.RadioButtons(rax, ('logistic', 'norm', 'cauchy'), active=0)
    def colorfunc(label):
        global function
        function = getattr(st,label)
        update(None)
        fig.canvas.draw_idle()
    colorradio.on_clicked(colorfunc)

    rax = plt.axes([0.025, 0.75, 0.15, 0.15], axisbg=axcolor)
    squareradio = wdg.RadioButtons(rax, ('x', 'x^2'), active=0)
    def squarefunc(label):
        global x
        if label == 'x':
            x = _x
        elif label == 'x^2':
            x = _x2
        update(None)
        fig.canvas.draw_idle()
    squareradio.on_clicked(squarefunc)

    plt.show()