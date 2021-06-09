from figure_basics import *
import numba

import scipy.signal
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mp

import pyafos

mp.rc('lines', lw=4)
mp.rc('font', size=60)


@numba.njit
def get_input(t, freq, amp, phase):
    out = np.zeros_like(t)
    for i in range(freq.shape[0]):
        out += amp[i] * np.sin(freq[i]*t + phase[i])
    return out

@numba.njit
def get_pool_output(phi, amp):
    out = np.zeros_like(phi[0,:])
    for i in range(phi[0,:].size):
        for j in range(phi[:,0].size):
            out[i] += amp[j,i] * np.cos(phi[j,i])
    return out
    
def plot_results(frequencies, amplitudes, omega, alpha, in_signal, output, N, ind):
    fig = plt.figure(figsize=(19.8875,  15.9125))
    m = plt.get_current_fig_manager()
    m.resize(1591, 1273)

    ax2 = fig.add_subplot(111)
    ax2.set_xlim([0,t[-1]])
    ax2.set_ylim([0,55])
    ax2.set_xlabel('t')
    ax2.set_ylabel(r'$\displaystyle \omega$', size=80)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    
    ax2_subax1 = create_subax(fig, ax2, [0.2,0.1,0.3,0.3], 
                              xlimit=[0.1,.6], ylimit=[39,49],
                              xticks=[0.1,0.6], yticks=[40,44,48], side='r',
                              )
    ax2_subax2 = create_subax(fig, ax2, [0.4,0.65,0.3,0.3], 
                              xlimit=[3.5,4.], ylimit=[29.5,31],
                              xticks=[3.5,4.], yticks=[30,31], side='r',
                              )
    
    ax2_subax3 = create_subax(fig, ax2, [0.6,0.1,0.3,0.3], 
                          xlimit=[15.0,15.5], ylimit=[29.8,30.2],
                          xticks=[15.0,15.5], yticks=[30,30.2], side='b',
                          )
    axes_plot = [ax2, ax2_subax1, ax2_subax2, ax2_subax3]
    
    for a in axes_plot:
        for i in range(N):
            a.plot(t, omega[i,:], lw=6)
            a.plot([t[0],t[-1]], [frequencies[i], frequencies[i]], 'k--', lw=6)
        if a != ax2:
            for i in ind:
                a.plot([t[i], t[i]], [0,50], '--k', lw=2)
    
    fig.savefig('pool_one_oscill_om.pdf', bbox_inches='tight')


    fig = plt.figure(figsize=(19.8875,  15.9125))
    m = plt.get_current_fig_manager()
    m.resize(1591, 1273)

    ax2 = fig.add_subplot(111)
    ax2.set_xlim([0,t[-1]])
    ax2.set_ylim([0,2.1])
    ax2.set_xlabel('t')
    ax2.set_ylabel(r'$\displaystyle \alpha$', size=80)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    
    ax2_subax1 = create_subax(fig, ax2, [0.2,0.05,0.3,0.3], 
                              xlimit=[0.1,.6], ylimit=[0.2,1.],
                              xticks=[0.1,0.6], yticks=[0.2,0.6,1.], side='r',
                              )
    ax2_subax2 = create_subax(fig, ax2, [0.25,0.45,0.3,0.3], 
                              xlimit=[3.5,4.], ylimit=[1.78,1.84],
                              xticks=[3.5,4.], yticks=[1.78,1.84], side='b',
                              )
    
    ax2_subax3 = create_subax(fig, ax2, [0.65,0.5,0.3,0.3], 
                          xlimit=[15.0,15.5], ylimit=[1.98,1.983],
                          xticks=[15.0,15.5], yticks=[1.98,1.983], side='b',
                          )
    axes_plot = [ax2, ax2_subax1, ax2_subax2, ax2_subax3]
    
    for a in axes_plot:
        for i in range(N):
            a.plot(t, alpha[i,:], lw=6)
            a.plot([t[0],t[-1]], [amplitudes[i], amplitudes[i]], 'k--', lw=6)
        if a != ax2:
            for i in ind:
                a.plot([t[i], t[i]], [0,50], '--k', lw=2)
    fig.savefig('pool_one_oscill_alpha.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(19.8875,  15.9125))
    m = plt.get_current_fig_manager()
    m.resize(1591, 1273)

    ax2 = fig.add_subplot(111)
    ax2.set_xlim([0,t[-1]])
    ax2.set_ylim([0,4.])
    ax2.set_xlabel('t')
    ax2.set_ylabel(r'$\displaystyle |I(t) - \alpha\cos\phi|^2$', size=80)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(40)
    
    ax2_subax1 = create_subax(fig, ax2, [0.15,0.65,0.3,0.3], 
                              xlimit=[0.1,.6], ylimit=[-0.01,3],
                              xticks=[0.1,0.6], yticks=[0.,1,3.], side='r',
                              )
    ax2_subax2 = create_subax(fig, ax2, [0.25,0.1,0.3,0.3], 
                              xlimit=[3.5,4.], ylimit=[-0.0001,0.05],
                              xticks=[3.5,4.], yticks=[0,0.05], side='b',
                              )
    
    ax2_subax3 = create_subax(fig, ax2, [0.65,0.1,0.3,0.3], 
                          xlimit=[15.0,15.5], ylimit=[-0.0001,0.002],
                          xticks=[15.0,15.5], yticks=[0,0.002], side='b',
                          )
    axes_plot = [ax2, ax2_subax1, ax2_subax2, ax2_subax3]
    
    for a in axes_plot:
        for i in range(N):
            a.plot(t, (in_signal-output)**2, lw=6)
        if a != ax2:
            for i in ind:
                a.plot([t[i], t[i]], [0,50], '--k', lw=2)
    fig.savefig('pool_one_oscill_F.pdf', bbox_inches='tight')
    # fig = plt.figure(3)
    # plt.plot(t, (in_signal-output)**2)
    # plt.xlim([1, 1.5])
    # plt.ylim([-0.1, 1.5])
    # plt.ylabel(r'$|I(t)|^2$')
    # plt.xlabel('Time [s]')
    
def find_zero_switching(sig):
    l = [0]
    found = False
    for i, o in enumerate(sig):
        if o < 10e-5 and not found and (i - l[-1] > 10):
            found = True
            l.append(i)
        elif o > 10e-5 and found and (i - l[-1] > 10):
            found = False
            l.append(i)
    return l[1:]

def simulate_pool(N, K, lamb, eta, amplitudes, frequencies, phases):
    dt = 10**-5 #need 10-7 for Euler with K = 10*7
    save_dt = 10**-3
    t_end = 21. #50 * 2 * np.pi / omegaC#30.
    t_start = 0.

    oscill = pyafos.PoolAFO()
    oscill.initialize(N, K, lamb, eta)
    oscill.input().vec_of_sines(frequencies, amplitudes, phases)

    phi0 = np.zeros([N])
    alpha0 = np.ones([N]) * 0
    omega0 = np.array([50.])/lamb #np.array([40., 69., 71.])/lamb
    x0 = np.hstack([phi0, omega0, alpha0])
    pyafos.integrate(oscill, t_start,t_end,x0,dt,save_dt)

    #generate data to be plotted    
    t = oscill.t()
    phi = oscill.y()[0:N,:]
    omega = lamb*oscill.y()[N:2*N,:]
    alpha = oscill.y()[2*N:,:]
    
    return t, phi, omega, alpha

amplitudes = np.array([2.])
frequencies = np.array([30.])
# frequencies = np.array([30., 45., 60.])
phases = np.zeros([3])

N = 1

lamb = 1
eta = 2.

K = 100000
t, phi, omega, alpha = simulate_pool(N, K, lamb, eta, amplitudes, frequencies, phases)
in_signal = get_input(t, frequencies, amplitudes, phases)

output = get_pool_output(phi, alpha)

ind = find_zero_switching((in_signal-output)**2)

plot_results(frequencies, amplitudes, omega, alpha, in_signal, output, N, ind)



plt.show()