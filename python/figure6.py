from figure_basics import *

import pyafos

from analysis_utils import find_roots, compute_maps

use_saved_data = False

mp.rc('lines', lw=4)
mp.rc('font', size=60)
    
K = 10.**7
omegaF = 30.
freq = omegaF * np.array([1.,sqrt(2.),pi/sqrt(2.)]) 
amp = np.array([1.3,1.,1.4]) 
phase = np.array([0.,0.,0.]) + np.pi/2.0
lamb = 1
dt = 10**-8
save_dt = 10**-3
t_end = 30.
t_start = 0.
omega0 = 20./lamb
phi0 = 0.

if use_saved_data:
    # load data from file
    with open('figure6_data.npy', 'rb') as f:
        t = np.load(f)
        phi = np.load(f)
        omega = np.load(f)
        roots = np.load(f)
else:
    #run an integration
    oscill = pyafos.PhaseAFO()
    oscill.initialize(K, lamb)
    oscill.input().vec_of_sines(freq, amp, phase)
    pyafos.integrate(oscill, t_start,t_end,np.array([phi0,omega0]),dt,save_dt)

    #generate data to be plotted    
    t = oscill.t()
    phi = oscill.y()[0,:]
    omega = oscill.y()[1,:]

    roots = find_roots(t, freq, amp, phase)

    # save data
    with open('figure6_data.npy', 'wb') as f:
        np.save(f, t)
        np.save(f, phi)
        np.save(f, omega)
        np.save(f, roots)

roots_corrected = roots + (2.*pi/omegaF - roots[-1])
n_roots = size(roots)
print('num roots is ',n_roots)
#print 'roots ', roots
#print 'roots corrected', roots_corrected

omega_p, omega_m = compute_maps(roots_corrected, omega0*exp(-lamb*roots[0])+pi, omega0*exp(-lamb*roots[0]), lamb)    

#plot stuff
fig = plt.figure(figsize=(19.8875,  15.9125))

m = plt.get_current_fig_manager()
m.resize(1591, 1273)


#axes 1
ax2 = fig.add_subplot(111)
ax2.set_xlim([0,t_end])
ax2.set_ylim([0,70])
ax2.set_xlabel('t')
ax2.set_ylabel(r'$\displaystyle \omega$', size=80)

# for tick in ax2.xaxis.get_major_ticks():
#     tick.label.set_fontsize(40)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(40)

    ax2_subax1 = create_subax(fig, ax2, [0.67,0.1,0.3,0.5], 
                              xlimit=[26,27], ylimit=[51,65],
                              xticks=[26.0,26.5,27.0], yticks=[55,60,65], side='b',
                              )
    ax2_subax2 = create_subax(fig, ax2, [0.25,0.1,0.3,0.5], 
                              xlimit=[2.0,3.], ylimit=[36,50],
                              xticks=[2.0,2.5,3.0], yticks=[40,45,50], side='r',
                              )
    axes_plot = [ax2, ax2_subax1, ax2_subax2]
# axes_plot = [ax2]

for a in axes_plot:
    if a == ax2:
        mw = 2.0
        ms = 4.0
    else:
        mw = 4.0
        ms = 10.0

        #plot cos = 0
        tx = arange((pi/2)/omegaF,t_end,pi/omegaF   )
        for i in tx:
            a.plot([i,i],[0,110],ls=':',lw=2,color='k')

    a.plot(t, lamb*omega, color='b', lw=6, ls='-')                    
    a.plot(roots, lamb*omega_p, ls='', marker='x', markeredgewidth=mw, markersize=ms, color='r')
    a.plot(roots, lamb*omega_m, ls='', marker='x', markeredgewidth=mw, markersize = ms, color='g')

plt.show()