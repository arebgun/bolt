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

    def angle_diff(angles1, angle2):
        return np.abs((angles1-angle2+np.pi)%(2*np.pi)-np.pi)

    def angle_diff2(angles1, angle2):
        return np.abs((angles1-angle2+180)%(360)-180)

    def decay_envelope(x, mu, sigma):
        return np.exp(-angle_diff2(x,mu)/(2*sigma))

    def von_mises_function(x, kappa, mu):
        return np.exp(kappa*(np.cos(np.radians(x)-np.radians(mu))-1))

    fig, (ax1, ax2) = plt.subplots(2,1)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    # x = np.arange(-np.pi, np.pi, 0.01)
    x = np.arange(-180, 180, 0.1)
    kappa0 = 6.66
    mu0 = 0
    sigma0=8
    f1 = decay_envelope(x,mu0,sigma0)
    l1, = ax1.plot(x, f1, lw=2, color='red')
    # ax1.axis([-np.pi, np.pi, 0, 1])
    ax1.axis([-180, 180, 0, 1])
    # f2 = st.vonmises.pdf(np.radians(x), kappa0, loc=np.radians(mu0))/\
    #     st.vonmises.pdf(np.radians(mu0), kappa0, loc=np.radians(mu0))
    f2 = von_mises_function(x, kappa0, mu0)
    l2, = ax2.plot(x, f2, lw=2, color='red')
    # ax2.axis([-np.pi, np.pi, -np.pi, np.pi])
    ax2.axis([-180, 180, 0, 1])


    axcolor = 'lightgoldenrodyellow'
    axkappa = plt.axes([0.25, 0.17, 0.65, 0.03], axisbg=axcolor)
    axmu = plt.axes([0.25, 0.13, 0.65, 0.03], axisbg=axcolor)
    axsigma = plt.axes([0.25, 0.09, 0.65, 0.03], axisbg=axcolor)

    # smu = wdg.Slider(axmu, 'mu', -np.pi, np.pi, valinit=mu0)
    skappa = wdg.Slider(axkappa, 'kappa', 0.0001, 100, valinit=kappa0)
    smu = wdg.Slider(axmu, 'mu', -180, 180, valinit=mu0)
    ssigma = wdg.Slider(axsigma, 'sigma', 0.0001, 100, valinit=sigma0)

    def update(val):
        kappa = skappa.val
        mu = smu.val
        sigma = ssigma.val
        f1 = decay_envelope(x, mu=mu, sigma=sigma)
        l1.set_ydata(f1)
        # l2.set_ydata(angle_diff2(x,mu))
        # f2 = st.vonmises.pdf(np.radians(x), kappa, loc=np.radians(mu))/\
        #      st.vonmises.pdf(np.radians(mu), kappa, loc=np.radians(mu))
        f2 = von_mises_function(x, kappa, mu)
        l2.set_ydata(f2)
        fig.canvas.draw_idle()
    skappa.on_changed(update)
    smu.on_changed(update)
    ssigma.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = wdg.Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    def reset(event):
        smu.reset()
        ssigma.reset()
    button.on_clicked(reset)

    # rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
    # colorradio = wdg.RadioButtons(rax, ('logistic', 'norm', 'cauchy'), active=0)
    # def colorfunc(label):
    #     global function
    #     function = getattr(st,label)
    #     update(None)
    #     fig.canvas.draw_idle()
    # colorradio.on_clicked(colorfunc)

    # rax = plt.axes([0.025, 0.75, 0.15, 0.15], axisbg=axcolor)
    # squareradio = wdg.RadioButtons(rax, ('x', 'x^2'), active=0)
    # def squarefunc(label):
    #     global x
    #     if label == 'x':
    #         x = _x
    #     elif label == 'x^2':
    #         x = _x2
    #     update(None)
    #     fig.canvas.draw_idle()
    # squareradio.on_clicked(squarefunc)

    plt.show()