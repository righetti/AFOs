from pyafos import afos
from pylab import *
from scipy import *
from numpy import *


K =  10000.
omegaF = 30
lamb = 1.
dt = 0.00001
save_dt = 0.001
t_end = 20.
res = afos.integrate_afo(0.,t_end,K,lamb,array([0.,10]),array([omegaF]),array([1.]),array([0.]),dt, save_dt)
t = res[0,:]
phi = res[1,:]
omega = res[2,:]

subplot(2,1,1)
plot(t, phi)
subplot(2,1,2)
plot(t, omega)

om = omega[10000:]
print (lamb*min(om) - omegaF)/lamb 
print (lamb*max(om) - omegaF)/lamb
print (lamb*mean(om) - omegaF)/lamb