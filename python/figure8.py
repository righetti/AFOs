# numba significantly speeds up code
import numba

import pyafos
import IPython
import scipy.signal
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mp

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
    
mp.rc('lines', lw=12)
mp.rc('savefig', format='pdf')
mp.rc('font', size = 80)
mp.rc('text', usetex = True)
mp.rc('figure', figsize=(19.8875,  2*15.9125))

# mp.rcdefaults()

def plot_results(frequencies, amplitudes, omega, alpha, in_signal, output, N):
    fig = plt.figure()
    plt.subplot(3,1,1)
    for i in range(N):
        plt.plot(t, omega[i,:])
        plt.plot([t[0],t[-1]], [frequencies[i], frequencies[i]], 'k--')
    plt.xlim([0, 50.])
    plt.ylim([-10, 90.])
    plt.ylabel(r'$\omega_i$')

    plt.subplot(3,1,2)
    for i in range(N):
        plt.plot(t, alpha[i,:])
        plt.plot([t[0],t[-1]], [amplitudes[i], amplitudes[i]], 'k--')
    plt.xlim([0, 50.])
    plt.ylim([0, 2.])
    plt.ylabel(r'$\alpha_i$')

    plt.subplot(3,1,3)
    plt.plot(t, (in_signal-output)**2)
    plt.xlim([0, 50.])
    plt.ylim([0, 12.])
    plt.ylabel(r'$|I(t)|^2$')
    plt.xlabel('Time [s]')
    fig.savefig('pool_3oscill_K'+str(K)+'.pdf', bbox_inches='tight')
    
    

def simulate_pool(N, K, lamb, eta, amplitudes, frequencies, phases):
    dt = 10**-5 #need 10-7 for Euler with K = 10*7
    save_dt = 10**-3
    t_end = 50. #50 * 2 * np.pi / omegaC#30.
    t_start = 0.

    oscill = pyafos.PoolAFO()
    oscill.initialize(N, K, lamb, eta)
    oscill.input().vec_of_sines(frequencies, amplitudes, phases)

    phi0 = np.zeros([3])
    alpha0 = np.zeros([3])
    omega0 = np.array([40., 69., 71.])/lamb
    x0 = np.hstack([phi0, omega0, alpha0])
    pyafos.integrate(oscill, t_start,t_end,x0,dt,save_dt)

    #generate data to be plotted    
    t = oscill.t()
    phi = oscill.y()[0:3,:]
    omega = lamb*oscill.y()[3:6,:]
    alpha = oscill.y()[6:,:]
    
    return t, phi, omega, alpha




amplitudes = np.array([1.3, 1., 1.4])
frequencies = np.array([30., 30.*np.sqrt(2), 30.*np.pi/np.sqrt(2)])
# frequencies = np.array([30., 45., 60.])
phases = np.zeros([3])


N = 3

lamb = 1
eta = 10

plt.close('all')

K = 10
t, phi, omega, alpha = simulate_pool(N, K, lamb, eta, amplitudes, frequencies, phases)
in_signal = get_input(t, frequencies, amplitudes, phases)
output = get_pool_output(phi, alpha)
plot_results(frequencies, amplitudes, omega, alpha, in_signal, output, N)

K = 100
t, phi, omega, alpha = simulate_pool(N, K, lamb, eta, amplitudes, frequencies, phases)
output = get_pool_output(phi, alpha)
plot_results(frequencies, amplitudes, omega, alpha, in_signal, output, N)

K = 10000
t, phi, omega, alpha = simulate_pool(N, K, lamb, eta, amplitudes, frequencies, phases)
output = get_pool_output(phi, alpha)
plot_results(frequencies, amplitudes, omega, alpha, in_signal, output, N)

