from figure_basics import *

import pyafos

K = 20
omegaF = 60.
lamb = 1.
dt = 10**-6
save_dt = 10**-4
t_end = 20.
t_start = 0.
omega0 = 90./lamb
phi0 = 0.


#run an integration
oscill = pyafos.PhaseAFO()
oscill.initialize(K, lamb)
oscill.input().vec_of_sines(np.array([omegaF]),np.array([1.0]),np.array([np.pi/2.]))
oscill.integrate(t_start,t_end,np.array([phi0,omega0]),dt,save_dt)

#get the data to be plotted    
t = oscill.t()
phi = oscill.y()[0,:]
omega = oscill.y()[1,:]

#plot results
fig = plt.figure(figsize=(7.5*1.5, 7.5))

plt.plot(t, omega)
plt.plot(t, np.ones(np.size(t))*72, ls='--')

plt.xlim(0,20)
plt.ylim(49,91)

#put the text    
plt.ylabel(r'$\displaystyle \omega$', fontsize=30)
plt.xlabel(r'Time[s]', fontsize=30)


# m = plt.get_current_fig_manager()
# m.resize(1591, 1273)

plt.subplots_adjust(left=0.14, bottom=0.18, top=0.97, right=0.92)


plt.show()