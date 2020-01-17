#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 01:23:41 2018

@author: righetti

code to generate data and plots for sweep tests of the afos tracking changing
frequencies
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from scipy.interpolate import interp1d

import pickle

from pyafos import PhaseAFO

from numpy.ma.core import exp
from numpy import sqrt, arange, cos, sin, pi, meshgrid, size, zeros
from scipy.signal import hilbert


fig_width_pt = 4*246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size = [fig_width, fig_height]


# params = {'backend': 'ps',
#           'axes.labelsize': 40,
#           'font.size': 40,
#           'legend.fontsize': 40,
#           'xtick.labelsize': 40,
#           'ytick.labelsize': 40,
#           'lines.linewidth': 6,
#           'text.usetex': True,
#           'figure.figsize': fig_size}

# mp.rc('lines', lw=6)
# mp.rc('savefig', format='pdf')
# mp.rc('font', size=40)
# mp.rc('text', usetex=True)


def find_min_max(omega):
    """ this function finds the min and maxes of the signal assuming
        that the signal reach a max and then goes monotonously to a min
    """
    current_max = []
    current_min = []

    if(omega[0] < omega[1]):
        seeking_max = True
    else:
        seeking_max = False

    old_om = omega[0]

    for i, o in enumerate(omega[1:]):
        if seeking_max:
            if o < old_om:
                current_max.append(i)
                seeking_max = False
        else:
            if o > old_om:
                current_min.append(i)
                seeking_max = True
        old_om = o

    # make things the same length
    if len(current_max) > len(current_min):
        return current_min, current_max[0:len(current_min)]
    else:
        return current_min[0:len(current_max)], current_max


def plot_sweep_afosimple(use_saved_data):
    mp.rc('lines', lw=4)
    #mp.rc('font', size=60)

    K = 10000.
    omegaF = 10000.
    num_data_point = 10
    omegaC = np.logspace(-2, 2, num=num_data_point)
    lamb = 1.
    dt = 10**-4
    save_dt = 10**-4
    omega0 = omegaF
    phi0 = 0

    # run an integration and generate data to be plotted
    data_filename = 'plot_sweep_afosimple.pkl'
    if not use_saved_data:
        amp = np.zeros(num_data_point)
        phase = np.zeros(num_data_point)
        for i, omC in enumerate(omegaC):
            t_end = 2*10.*2*pi/omC
            t_start = 0.
            if omC < 1e-1:
                save_dt = 1e-1
                dt = 10**-4
            else:
                if omC < 1e1:
                    save_dt = 1e-3
                    dt = 10**-5
                else:
                    save_dt = 1e-5
                    dt = 10**-5
            print(i, omC, t_end)
            afos = PhaseAFO()
            afos.initialize_frequency_changing_sine(K, omegaF, omC, lamb)
            afos.integrate(t_start, t_end, np.array([phi0, omega0]),
                           dt, save_dt)
            t = afos.t()
            omega = afos.y()[1, :]
            mines, maxes = find_min_max(omega)
            f_min = interp1d(t[mines], omega[mines])
            f_max = interp1d(t[maxes], omega[maxes])

            if(mines[0] > maxes[0]):
                t_play = t[mines[0]:maxes[-1]]
            else:
                t_play = t[maxes[0]:mines[-1]]

            f_H = hilbert((f_max(t_play) + f_min(t_play))/2. - omegaF)
            H2 = hilbert(cos(omC*t_play))

            f_H_envelop = np.abs(f_H/H2)
            instantaneous_phase = np.angle(f_H_envelop)
            amp[i] = 20*np.log10(np.median(f_H_envelop))
            phase[i] = np.median(instantaneous_phase)
            print(amp[i])

        data_file = open(data_filename, 'wb')
        pickle.dump(amp, data_file)
        pickle.dump(phase, data_file)
        data_file.close()
    else:
        data_file = open(data_filename, 'rb')
        amp = pickle.load(data_file)
        phase = pickle.load(data_file)
        data_file.close()

    # plot stuff
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogx(omegaC, amp)
    plt.subplot(2, 1, 2)
    plt.semilogx(omegaC, phase)

#    m = plt.get_current_fig_manager()
#    m.resize(1591, 1273)

#   fig.set_size_inches([19.8875, 15.9125])


plot_sweep_afosimple(False)
