from pylab import *
from scipy import *
from numpy import *
import sys
import numpy
from test.test_colorsys import frange


def run_afo():
    dt = 0.00001
    t = arange(0,.1,dt)
    
    K=100000
    omegaF = 100
    
    y0 = array([0,150])
    
    my_afo = afo_ode(K,omegaF)
    y = euler_integration(my_afo, t, y0,dt)
    subplot(3,1,1)
    plot(t,y[0,:])
    subplot(3,1,2)
    plot(t,y[1,:],t,100+50*exp(-t),t,100+50*exp(-3*t))
    subplot(3,1,3)
    plot(t,cos(omegaF*t))

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
        dydt[0] = y[1] - self.K * sin(y[0]) * cos(self.omegaF*t)
        dydt[1] = - self.K * sin(y[0]) * cos(self.omegaF*t)
    
        return dydt