from pylab import *
from scipy import *
from numpy import *
import sys
import numpy
<<<<<<< HEAD
from test.test_colorsys import frange
from numpy.ma.core import exp
from mpl_toolkits.mplot3d import axes3d
=======
>>>>>>> aee62cccdc46865778ce8dbc578dc70877964b3e


def run_afo(Tend=0.1,K=10000,dt=0.0001):
    t = arange(0,Tend,dt)
    
    omegaF = 10
    
    y0 = array([0,10])
    
    my_afo = afo_ode(K,omegaF)
    y = euler_integration(my_afo, t, y0,dt)
#    subplot(3,1,1)
#    plot(t,y[0,:])
#    for i in frange(0,Tend,(1./omegaF)*(pi/2)):
#        vlines(i, 0, i*omegaF)
#    subplot(3,1,2)
#    for i in frange(0,Tend,(1./omegaF)*(pi/2)):
#        vlines(i, 130, 150)
    #plot(t,y[1,:])
    plot(mod(t+pi/(2*omegaF),2*pi/omegaF)-pi/(2*omegaF),y[0,:]-y[1,:],zs=y[1,:],marker='o',ms=10)
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
        dydt[0] = y[1] - self.K * sin(y[0]) * cos(self.omegaF*t)
        dydt[1] = - self.K * sin(y[0]) * cos(self.omegaF*t)
    
        return dydt
    
def plot_cpp_res():
    res = loadtxt('result.txt')
    plot(res[:,0],res[:,2])
 
def plot_invariant_manifold():
    epsilon = 0.0001
    omegaF = 10
    theta_p = arange(-pi/2+0.01, pi/2-0.01, 0.01)
    theta_m = arange(pi/2+0.01, 3*pi/2-0.01, 0.01)
    theta_p = theta_p / omegaF
    theta_m = theta_m / omegaF
    Om = arange(-1,10,1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def invariant_k(k,ax):
        theta1p, Om1p = meshgrid(theta_p,Om)
        theta1m, Om1m = meshgrid(theta_m,Om)
        omega_p = (k*pi-Om1p)*(1+epsilon/(cos(k*pi)*cos(omegaF*theta1p))) 
        omega_m = (k*pi-Om1m)*(1+epsilon/(cos(k*pi)*cos(omegaF*theta1m)))
        if k%2: ##if k is odd
            ax.plot_surface(theta1p, Om1p, omega_p, color='b')
            ax.plot_surface(theta1m, Om1m, omega_m, color='r')
        else:
            ax.plot_surface(theta1p, Om1p, omega_p, color='r')
            ax.plot_surface(theta1m, Om1m, omega_m, color='b')
    
    invariant_k(0,ax)
    invariant_k(1,ax)
    invariant_k(2,ax)
    invariant_k(3,ax)
    invariant_k(4,ax)
    invariant_k(5,ax)
    
    

 
 
def plot_fixed_point():
    omegaF = arange(0,100,0.001)
    fixed = pi / (1-exp(-pi/omegaF))
    plot(omegaF,fixed-omegaF,omegaF,pi/2*ones(size(omegaF)),omegaF,pi*ones(size(omegaF)),omegaF,zeros(size(omegaF)))
 
def stupid_test():
    omega0 = 10;
    omegaF = 100;
    
    iter = 300
    omega = zeros(2*iter)
    omega[0] = omega0
    for i in range(iter-1):
            omega[2*i+1] = omega[2*i]*exp(-(pi/omegaF))
            omega[2*i+2] = omega[2*i+1] + pi
#        else:
#            omega[2*i+1] = omega[2*i] + omega[2*i]*(1-exp(-pi/omegaF))
#            omega[2*i+2] = omega[2*i+1] - pi
    
#    subplot(3,1,2)
    plot(arange(iter)*pi/omegaF, omega[::2], 'x',arange(iter)*pi/omegaF+pi/omegaF, omega[1::2], 'x')
