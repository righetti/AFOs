from afos import afos
from pylab import *
from scipy import *
from numpy import *


K =  10000
omegaF = 100
lamb = 10
dt = 0.000001
save_dt = 0.001
t_end = 200
a = afos.PhaseAFO(K,omegaF,lamb)
res = afos.integrate_afo(a,0,t_end,array((0.,10/lamb)),dt, save_dt)
t = res[0,:]
alpha = res[1,:]
omega = res[2,:]

plot(t, lamb*omega)
show()

om = omega[1000:]
print lamb*min(om) - omegaF 
print lamb*max(om) - omegaF 
print lamb*mean(om) - omegaF