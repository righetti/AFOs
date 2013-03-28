from pylab import *
from scipy import *
from numpy import *
import sys
import numpy
from test.test_colorsys import frange


def run_afo(Tend=0.1,K=1000,dt=0.0001):
    t = arange(0,Tend,dt)
    
    omegaF = 100
    
    y0 = array([0,100])
    
    my_afo = afo_ode(K,omegaF)
    y = euler_integration(my_afo, t, y0,dt)
#    subplot(3,1,1)
#    plot(t,y[0,:])
#    for i in frange(0,Tend,(1./omegaF)*(pi/2)):
#        vlines(i, 0, i*omegaF)
#    subplot(3,1,2)
#    for i in frange(0,Tend,(1./omegaF)*(pi/2)):
#        vlines(i, 130, 150)
    plot(t,y[1,:])
#    subplot(3,1,3)
#    plot(t,cos(omegaF*t))
#    for i in frange(0,Tend,(1./omegaF)*(pi/2)):
#        vlines(i, -1, 1)
#    hlines(0,0,Tend)
#    xlim([0,Tend])

def euler_integration(ode,t,y0,dt=0.001):
    
    y = zeros((size(y0),size(t)))
    y[:,0] = y0
    for i in range(0,size(t)-1):
        y[:,i+1] = y[:,i] + ode.dydt(t[i],y[:,i]) * dt
    
    return y
        
    
class afo_ode:
    K = 0.0
    omegaF = 0.0
    def __init__(self,K,omegaF):
        self.K = K
        self.omegaF = omegaF
    
    def dydt(self,t,y):
        dydt = array([0,0])
        dydt[0] = y[1] - self.K * sin(y[0]) * cos(self.omegaF*t-pi/2)
        dydt[1] = - self.K * sin(y[0]) * cos(self.omegaF*t-pi/2)
    
        return dydt
    
    
def stupid_test():
    omega0 = 100;
    omegaF = 100;
    
    iter = 100
    omega = zeros(2*iter)
    omega[0] = omega0
    omega[1] = omega0 + pi
    for i in range(iter-1):
            omega[2*i+1] = omega[2*i]*exp(-(pi/omegaF))
            omega[2*i+2] = omega[2*i+1] + pi
#        else:
#            omega[2*i+1] = omega[2*i] + omega[2*i]*(1-exp(-pi/omegaF))
#            omega[2*i+2] = omega[2*i+1] - pi
    
#    subplot(3,1,2)
    plot(arange(iter)*pi/omegaF, omega[::2], 'x',arange(iter)*pi/omegaF, omega[1::2], 'x')