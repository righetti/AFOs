from figure_basics import *

import pyafos

from analysis_utils import find_roots, compute_maps

mp.rc('lines', lw=4)
mp.rc('font', size=60)

K = 10.**6
omegaF = 30.
freq = omegaF * np.array([1.,2.,3.])
amp = np.array([1.3,1.,1.4])
phase = np.array([0.4,0.0,1.3]) + np.pi/2.0
lamb = 1
dt = 10**-7
save_dt = 10**-3
t_end = 8.
t_start = 0.
omega0 = 20./lamb
phi0 = 0.

#run an integration
oscill = pyafos.PhaseAFO()
oscill.initialize(K, lamb)
oscill.input().vec_of_sines(freq,amp,phase)
oscill.integrate(t_start,t_end,np.array([phi0,omega0]),dt,save_dt)

#get the data to be plotted    
t = oscill.t()
phi = oscill.y()[0,:]
omega = oscill.y()[1,:]

deltaT = np.linspace(0.0,2*pi/omegaF,10000)
roots = find_roots(deltaT, freq, amp, phase)
roots_corrected = roots + (2.*pi/omegaF - roots[-1])
n_roots = size(roots)
print('num roots is ', n_roots)
print('roots ', roots)
print('roots corrected', roots_corrected)

#     omega_bar_p = omegaF * n_roots * (0.5 / lamb) + pi + omegaF * 0.5 * sum(roots[:-1])
#     omega_bar_m = omegaF * n_roots * (0.5 / lamb) + omegaF * 0.5 * sum(roots[:-1])
#     omega_bar_avg = omegaF * n_roots * (0.5 / lamb) + pi*0.5 + omegaF * 0.5 * sum(roots[:-1])
omega_bar_p = pi / (exp(lamb*2*pi/omegaF)-1) * (sum(exp(lamb*roots_corrected)))
omega_bar_m = pi / (exp(lamb*2*pi/omegaF)-1) * (1 + sum(exp(lamb*roots_corrected[:-1])))
omega_bar_avg = pi / (2*(exp(lamb*2*pi/omegaF)-1)) * (1 + exp(lamb*2*pi/omegaF) + 2*sum(exp(lamb*roots_corrected[:-1])))

print('omega_bar o+ and o-', omega_bar_avg * lamb, omega_bar_p, omega_bar_m)

print('approx omega_bar ', omegaF * n_roots/2 + lamb * pi / 2. + omegaF/2. * sum(lamb*roots_corrected[:-1]))

tn = np.array([])
for i in range(int(t_end / (2*pi/omegaF))+1):
    tn = np.append(tn, i*2*pi/omegaF + roots)
omega_p, omega_m = compute_maps(tn, omega0*exp(-lamb*roots[0])+pi, omega0*exp(-lamb*roots[0]), lamb)
print('omega_p omega_m', omega_p[-4:], omega_m[-4:])
omega_avg = (omega0 -n_roots/2.*omegaF) * exp(-lamb*t) + n_roots/2.*omegaF
tn = tn
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

for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(40)

ax2_subax1 = create_subax(fig, ax2, [0.67,0.1,0.3,0.5], 
                          xlimit=[6.5,6.9], ylimit=[60-4,60+4],
                          xticks=[6.5,6.7,6.9], yticks=[60-3,60,60+3], side='b',
                          )
ax2_subax2 = create_subax(fig, ax2, [0.25,0.1,0.3,0.5], 
                          xlimit=[0.15,0.55], ylimit=[25,37],
                          xticks=[0.2,0.35,0.5], yticks=[25,30,35], side='r',
                          )
axes_plot = [ax2, ax2_subax1, ax2_subax2]

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

    if a == ax2_subax1:
        a.plot(t, omega_bar_avg*np.ones(size(t)), ls='-.',lw=4,color='k')

    a.plot(t, lamb*omega, color='b', lw=6, ls='-')                    
    a.plot(tn, lamb*omega_p, ls='', marker='x', markeredgewidth=mw, markersize=ms, color='r')
    a.plot(tn, lamb*omega_m, ls='', marker='x', markeredgewidth=mw, markersize = ms, color='g')
    a.plot(t, lamb*omega_avg, ls='--',lw=4, color='k')

plt.show()