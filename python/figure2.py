from figure_basics import *

import pyafos

K = 10.**7
omegaF = 100.
lamb = 1.
dt = 10**-7
save_dt = 10**-4
t_end = 6.
t_start = 0.
omega0 = 20./lamb
phi0 = 0.


#run an integration
oscill = pyafos.PhaseAFO()
oscill.initialize_vec_of_sines(K,np.array([omegaF]),np.array([1.0]),np.array([0.0]),lamb)
oscill.integrate(t_start,t_end,np.array([phi0,omega0]),dt,save_dt)

#get the data to be plotted    
t = oscill.t()
phi = oscill.y()[0,:]
omega = oscill.y()[1,:]


mp.rc('lines', lw=4)
mp.rc('font', size=60)

omega_bar_p = pi / (1 - exp(-lamb*pi/omegaF))
omega_bar_m = pi / (exp(lamb*pi/omegaF) - 1)
omega_bar_avg = 0.5*(omega_bar_p + omega_bar_m)

# print the \bar{\omega}
print('the average omega_bar between omegaM and omegaP (fixed points) is: ' + str(omega_bar_avg))

tn = arange(0,t_end,pi/omegaF)
omega_p = (omega[(np.abs(t-pi/omegaF/2)).argmin()]+pi - omega_bar_p)*exp(-lamb * tn) + omega_bar_p
omega_m = (omega[(np.abs(t-pi/omegaF/2)).argmin()] - omega_bar_m)*exp(-lamb * tn) + omega_bar_m
omega_avg = (omega0 -omega_bar_avg) * exp(-lamb*t) + omega_bar_avg

phi_m = zeros(size(tn))
phi_m[0] = phi[(np.abs(t-pi/omegaF/2)).argmin()]
for i in range(1,size(tn)):
    phi_m[i] = phi_m[i-1] + pi
phi_p = phi_m+pi


#plot results
fig = plt.figure(figsize=(19.8875,  15.9125))

m = plt.get_current_fig_manager()
m.resize(1591, 1273)

#axes 1
ax1 = fig.add_subplot(211)
ax1.set_xlim([0, t_end])
ax1.set_ylim([-10, 650])
ax1.set_ylabel(r'$\displaystyle \phi$', size=80)

for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(40)

ax1_subax1 = create_subax(fig, ax1, [0.68,0.1,0.3,0.5], 
                          xlimit=[5.55,5.75], ylimit=[550,580],
                          xticks=[5.6,5.7], yticks=[550,560,570], side='b',
                          )

ax1_subax2 = create_subax(fig, ax1, [0.05,0.49,0.3,0.5], 
                          xlimit=[0.15,0.35], ylimit=[10,40],
                          xticks=[0.2,0.3], yticks=[15,25,35], side='t',
                          )
axes_plot = [ax1, ax1_subax1, ax1_subax2]
for a in axes_plot:
    if a == ax1:
        mw = 2.0
        ms = 4.0
    else:
        mw = 4.0
        ms = 10.0

        #plot cos = 0
        tx = arange((pi/2)/omegaF,t_end,pi/omegaF   )
        for i in tx:
            a.plot([i,i],[0,650],ls=':',lw=2,color='k')

    a.plot(t, phi, color='b', lw=6, ls='-')
    a.plot(tn+pi/omegaF/2, phi_p, ls='', marker='x', markeredgewidth=mw, markersize=ms, color='r')
    a.plot(tn+pi/omegaF/2, phi_m, ls='', marker='x', markeredgewidth=mw, markersize=ms, color='g')


ax2 = fig.add_subplot(212)
ax2.set_xlim([0,t_end])
ax2.set_ylim([-10,110])
ax2.set_xlabel('t')
ax2.set_ylabel(r'$\displaystyle \omega$', size=80)

for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(40)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(40)

ax2_subax1 = create_subax(fig, ax2, [0.67,0.1,0.3,0.5], 
                          xlimit=[5.55,5.75], ylimit=[omega_bar_avg-3,omega_bar_avg+2.5],
                          xticks=[5.6,5.7], yticks=[100-3,100,100+3], side='b',
                          )
ax2_subax2 = create_subax(fig, ax2, [0.2,0.1,0.3,0.5], 
                          xlimit=[0.15,0.35], ylimit=[27,47],
                          xticks=[0.2,0.3], yticks=[30,35,40,45], side='r',
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
    a.plot(tn+pi/omegaF/2, lamb*omega_p, ls='', marker='x', markeredgewidth=mw, markersize=ms, color='r')
    a.plot(tn+pi/omegaF/2, lamb*omega_m, ls='', marker='x', markeredgewidth=mw, markersize = ms, color='g')
    a.plot(t, lamb*omega_avg, ls='--',lw=4, color='k')

plt.show()