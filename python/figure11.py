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
    
def simulate_pool(N, K, lamb, eta, amplitudes, frequencies, phases):
    dt = 10**-6 #need 10-7 for Euler with K = 10*7
    save_dt = 10**-3
    t_end = 200. #50 * 2 * np.pi / omegaC#30.
    t_start = 0.

    oscill = pyafos.PoolAFO()
    oscill.initialize(N, K, lamb, eta)
    oscill.input().vec_of_sines(frequencies, amplitudes, phases)

    phi0 = np.zeros([N])
    alpha0 = np.zeros([N])
#     omega0 = np.array([40., 69., 71.])/lamb
    omega0 = np.random.uniform(20,80,N)/lamb
    x0 = np.hstack([phi0, omega0, alpha0])
    pyafos.integrate(oscill, t_start,t_end,x0,dt,save_dt)

    #generate data to be plotted    
    t = oscill.t()
    phi = oscill.y()[0:N,:]
    omega = lamb*oscill.y()[N:2*N,:]
    alpha = oscill.y()[2*N:,:]
    
    return t, phi, omega, alpha



save_data = False

if save_data:
    amplitudes = np.array([1.3, 1., 1.4])
    frequencies = np.array([30., 30.*np.sqrt(2), 30.*np.pi/np.sqrt(2)])
    phases = np.zeros([3])
    
    N = 50

    lamb = 0.1
    eta = 1.

    K = 10000
    t, phi, omega, alpha = simulate_pool(N, K, lamb, eta, amplitudes, frequencies, phases)
    in_signal = get_input(t, frequencies, amplitudes, phases)

    output = get_pool_output(phi, alpha)

    # save data
    with open(f'pool_fig11_exp1.npy', 'wb') as f:
        np.save(f, t)
        np.save(f, phi)
        np.save(f, omega)
        np.save(f, alpha)
        np.save(f, in_signal)
        np.save(f, output)
        np.save(f, N)
        np.save(f, eta)
        np.save(f, lamb)
        np.save(f, K)
        np.save(f, amplitudes)
        np.save(f, frequencies)
        np.save(f, phases)
else:
    with open('pool_fig11_exp1.npy', 'rb') as f:
        t = np.load(f)
        phi = np.load(f)
        omega = np.load(f)
        alpha = np.load(f)
        in_signal = np.load(f)
        output = np.load(f)
        N = np.load(f)
        eta = np.load(f)
        lamb = np.load(f)
        K = np.load(f)
        amplitudes = np.load(f)
        frequencies = np.load(f)
        phases = np.load(f)


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
for i in range(len(frequencies)):
    plt.plot([t[0],t[-1]], [frequencies[i], frequencies[i]], 'k--')
plt.xlim([0,t[-1]])
plt.ylim([0,85])
plt.xticks([0,50,100,150,200])
plt.ylabel(r'$\omega_i$')

plt.subplot(2,1,2)
for i in range(N):
    plt.plot(t, alpha[i,:],':')
# for i in range(len(amplitudes)):
#     plt.plot([t[0],t[-1]], [amplitudes[i], amplitudes[i]], 'k--')
plt.xlim([0,t[-1]])
plt.ylabel(r'$\alpha_i$')
plt.xlabel('Time [s]')
plt.xticks([0,50,100,150,200])
# plt.ylim([-0.02,0.16])
fig.savefig('pool50_3freqs_variables.pdf', bbox_inches='tight')

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(t, (in_signal-output)**2)
plt.xlim([-1,t[-1]])
plt.ylim([0,3])
plt.xticks([0,50,100,150,200])
plt.ylabel(r'$|I(t) - \sum_i \alpha_i \cos\phi_i|^2$')

plt.subplot(2,1,2)
plt.plot(t, in_signal)
plt.xlim([-1,t[-1]])
plt.ylabel('I(t)')
plt.xticks([0,50,100,150,200])
plt.xlabel('Time [s]')
fig.savefig('pool50_3freqs_signal.pdf', bbox_inches='tight')


# compute the frequency / time plot

@numba.njit
def get_freq_map(t, phi, omega, alpha, delta_omega):
    bins = np.arange(10.,80.,delta_omega)
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


delta_omega = 1
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
plt.yticks(np.arange(0,80,20),['10','30','50','70'])
plt.xticks(np.arange(0,len(freq_time_map[:,0])+1,20000),['0','20','40','60','80','100','120','140','160','180','200'])
plt.colorbar(label='Amplitude', ticks=[0,1,1.4])

plt.plot([0,200000],[frequencies[0]-10,frequencies[0]-10],'r--',lw=4)
plt.plot([0,200000],[frequencies[1]-10,frequencies[1]-10],'r--',lw=4)
plt.plot([0,200000],[frequencies[2]-10,frequencies[2]-10],'r--',lw=4)
fig.savefig('pool50_3freqs_spectr.pdf', bbox_inches='tight')
