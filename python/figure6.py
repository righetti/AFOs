from figure_basics import *

import pyafos

from analysis_utils import compute_maps

def find_roots(t, f):
    ind = []
    for i in range(np.size(f)-1): 
        if(f[i]*f[i+1]<0):
            ind.append(i)
    return t[ind]

use_saved_data = False

mp.rc('lines', lw=4)
mp.rc('font', size=60)
    
K = 10.**6
lamb = 1
dt = 10**-7
save_dt = 10**-3
t_end = 50.
t_start = 0.
omega0 = 20./lamb
phi0 = 0.

use_saved_data = True

if use_saved_data:
    with open('figure6_data.npy', 'rb') as f:
        t = np.load(f)
        phi = np.load(f)
        omega = np.load(f)
        roots = np.load(f)
        x = np.load(f)
        y = np.load(f)
        z = np.load(f)
else:
    #run an integration
    oscill = pyafos.AfoLorentz()
    oscill.initialize(K, lamb)
    pyafos.integrate(oscill, t_start,t_end,np.array([phi0,omega0,1,1,1]),dt,save_dt)

    #get the data to be plotted    
    t = oscill.t()
    phi = oscill.y()[0,:]
    omega = oscill.y()[1,:]
    x = oscill.y()[2,:]
    y = oscill.y()[3,:]
    z = oscill.y()[4,:]

    roots = find_roots(t, z - 23)

    # save data
    with open('figure6bis_data.npy', 'wb') as f:
        np.save(f, t)
        np.save(f, phi)
        np.save(f, omega)
        np.save(f, roots)
        np.save(f, x)
        np.save(f, y)
        np.save(f, z)

omega_p, omega_m = compute_maps(roots, omega0*np.exp(-lamb*roots[0])+np.pi, omega0*np.exp(-lamb*roots[0]), lamb)

#plot stuff
fig = plt.figure(figsize=(19.8875,  15.9125))

m = plt.get_current_fig_manager()
m.resize(1591, 1273)


# axes 1
ax2 = fig.add_subplot(111)
ax2.set_xlim([0,t_end])
ax2.set_ylim([-0.1,25])
ax2.set_xlabel('t')
ax2.set_ylabel(r'$\displaystyle \omega$', size=80)

# for tick in ax2.xaxis.get_major_ticks():
#     tick.label.set_fontsize(40)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(40)

    ax2_subax1 = create_subax(fig, ax2, [0.67,0.66,0.3,0.3], 
                              xlimit=[40,44], ylimit=[5,12],
                              xticks=[40,42,44], yticks=[5,8,10], side='t',
                              )
    ax2_subax2 = create_subax(fig, ax2, [0.2,0.66,0.3,0.3], 
                              xlimit=[5.,9], ylimit=[-0.1,13],
                              xticks=[5.,7,9], yticks=[0,6,12], side='t',
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

        for i in roots:
            a.plot([i,i],[0,30],ls=':',lw=2,color='k')
    a.plot(t, lamb*omega.T, color='b', lw=6, ls='-')                    
    a.plot(roots, lamb*omega_p, ls='', marker='x', markeredgewidth=mw, markersize=ms, color='r')
    a.plot(roots, lamb*omega_m, ls='', marker='x', markeredgewidth=mw, markersize = ms, color='g')
    a.plot([0,t_end], [8.42, 8.42], 'k--')

plt.show()
fig.savefig('lorentz_omega.pdf', bbox_inches='tight')



#plot stuff
fig = plt.figure(figsize=(19.8875,  15.9125))

m = plt.get_current_fig_manager()
m.resize(1591, 1273)


# axes 1
ax2 = fig.add_subplot(211)
ax2.set_xlim([0,t_end])
# ax2.set_ylim([-0.1,25])
ax2.set_xlabel('t')
ax2.set_ylabel(r'$\displaystyle z$', size=80)

# for tick in ax2.xaxis.get_major_ticks():
#     tick.label.set_fontsize(40)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(40)

ax2.plot(t, z, color='b', lw=6, ls='-')                    

ax2 = fig.add_subplot(212)
ax2.set_xlim([0,t_end])
# ax2.set_ylim([-0.1,25])
ax2.set_xlabel('t')
ax2.set_ylabel(r'$\displaystyle \cos\phi$', size=80)

# for tick in ax2.xaxis.get_major_ticks():
#     tick.label.set_fontsize(40)
for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(40)

ax2.plot(t, cos(phi), color='b', lw=2, ls='-')
plt.show()
fig.savefig('lorentz_z.pdf', bbox_inches='tight')