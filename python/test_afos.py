from afos import afos
from pylab import *
from scipy import *
from numpy import *


K =  1000000
omegaF = 500
lamb = 500.
dt = 0.000001
save_dt = 0.001
t_end = 20
a = afos.PhaseAFO(K,omegaF,lamb)
res = afos.integrate_afo(a,0,t_end,array((0.,10/lamb)),dt, save_dt)
t = res[0,:]
alpha = res[1,:]
omega = res[2,:]

plot(t, lamb*omega)
show()

om = omega[10000:]
print (lamb*min(om) - omegaF)/lamb 
print (lamb*max(om) - omegaF)/lamb
print (lamb*mean(om) - omegaF)/lamb