import numba

import pyafos
import IPython
import scipy.signal
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mp

@numba.njit
def get_chirp_exp_input(t):
    return np.sin(200*t + 2*t*t) + np.sin(400*t - t*t*t/15) + np.sin(300*t)*np.exp(-(t-5)*(t-5)/2.5) + np.sin(400*t)*np.exp(-(t-30)*(t-30)/5.)

@numba.njit
def get_pool_output(phi, amp):
    out = np.zeros_like(phi[0,:])
    for i in range(phi[0,:].size):
        for j in range(phi[:,0].size):
            out[i] += amp[j,i] * np.cos(phi[j,i])
    return out
    


N = 100

K = 100
lamb = 10
eta = 0.5

dt = 10**-5 #need 10-7 for Euler with K = 10*7
save_dt = 10**-3
t_end = 40. #50 * 2 * np.pi / omegaC#30.
t_start = 0.

oscill = pyafos.PoolAFO()
oscill.initialize(N, K, lamb, eta)
oscill.input().chirps_and_exponentials()

phi0 = np.zeros([N])
alpha0 = np.zeros([N])
omega0 = np.random.uniform(50,500,N)/lamb
x0 = np.hstack([phi0, omega0, alpha0])
pyafos.integrate(oscill, t_start,t_end,x0,dt,save_dt)

#generate data to be plotted    
t = oscill.t()
phi = oscill.y()[0:N,:]
omega = lamb*oscill.y()[N:2*N,:]
alpha = oscill.y()[2*N:,:]


in_signal = get_chirp_exp_input(t)
output = get_pool_output(phi, alpha)

mp.rc('lines', lw=6)
mp.rc('savefig', format='pdf')
mp.rc('font', size = 80)
mp.rc('text', usetex = True)
mp.rc('figure', figsize=(19.8875,  2*15.9125))

plt.close('all')


fig = plt.figure()
plt.subplot(2,1,1)
for i in range(N):
    plt.plot(t, omega[i,:],':')
plt.plot([5,5],[0,500],'k--')
plt.plot([30,30],[0,500],'k--')
plt.plot([23.16,23.16],[0,500],'k--')
plt.xlim([0,t_end])
plt.ylim([0,500])
plt.ylabel(r'$\omega_i$')

plt.subplot(2,1,2)
for i in range(N):
    plt.plot(t, alpha[i,:],':')
plt.plot([5,5],[-1,1],'k--')
plt.plot([30,30],[-1,1],'k--')
plt.plot([23.16,23.16],[-1,1],'k--')
plt.xlim([0,t_end])
plt.ylabel(r'$\alpha_i$')
plt.xlabel('Time [s]')
plt.ylim([-0.02,0.16])
fig.savefig('time_var_variables.pdf', bbox_inches='tight')

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(t, (in_signal-output)**2)
plt.plot([5,5],[0,3],'k--')
plt.plot([30,30],[0,3],'k--')
plt.plot([23.16,23.16],[0,3],'k--')
plt.xlim([0,t_end])
plt.ylim([0,3])
plt.ylabel(r'$|I(t)|^2$')

plt.subplot(2,1,2)
plt.plot(t, in_signal)
plt.xlim([0,t_end])
plt.plot([5,5],[-3.5,3.5],'k--')
plt.plot([30,30],[-3.5,3.5],'k--')
plt.plot([23.16,23.16],[-3.5,3.5],'k--')
plt.xlim([0,t_end])
plt.ylim([-3.5,3.5])
plt.ylabel('F(t)')
plt.xlabel('Time [s]')
fig.savefig('time_var_signal.pdf', bbox_inches='tight')


# compute the frequency / time plot

@numba.njit
def get_freq_map(t, phi, omega, alpha, delta_omega):
    bins = np.arange(50.,500.,delta_omega)
    freq_time_map = np.zeros((t.size, bins.size))

    for i, tt in enumerate(t):
        for j,b in enumerate(bins):
            # find the freqs closer to b
            indices = np.asarray(np.abs(omega[:,i]-b)<delta_omega).nonzero()
            period = 2*np.pi/b
            time_test = np.linspace(0.,period,200)
            resp = np.zeros_like(time_test)
            for k in indices[0]:
                resp = resp + alpha[k,i] * np.cos(phi[k,i] + b*time_test)
            freq_time_map[i,j] = np.max(resp)
    return freq_time_map


delta_omega = 4
freq_time_map = get_freq_map(t,phi,omega,alpha, delta_omega)

mp.rc('lines', lw=6)
mp.rc('savefig', format='pdf')
mp.rc('font', size = 40)
mp.rc('text', usetex = True)
# mp.rc('figure', figsize=(19.8875,  15.9125))

plt.close('all')
fig = plt.figure(10,figsize=(20,10))
ax = plt.matshow(freq_time_map.T, fignum=10, aspect='auto', origin='lower')
plt.xlabel('Time[s]')
plt.ylabel(r'Frequency [rad$\cdot \textrm{s}^{-1}$]')
ax.axes.xaxis.set_ticks_position('bottom')
plt.yticks(np.arange(0,120,20),['0','100','200','300','400','500'])
plt.xticks(np.arange(0,40100,5000),['0','5','10','15','20','25','30','35','40'])
plt.colorbar(label='Amplitude')

plt.plot([5000,5000],[0,112],'r--')
plt.plot([30000,30000],[0,112],'r--')
plt.plot([23160,23160],[0,112],'r--')
fig.savefig('time_var_spectr.pdf', bbox_inches='tight')
