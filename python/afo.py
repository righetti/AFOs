import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mp

from afos import afos

from numpy.ma.core import exp
from mpl_toolkits.mplot3d import axes3d
from numpy import sqrt, arange, pi, meshgrid, cos, sin, mod, size, ones, zeros, linspace


fig_width_pt = 4*246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]


# params = {'backend': 'ps',
#           'axes.labelsize': 40,
#           'font.size': 40,
#           'legend.fontsize': 40,
#           'xtick.labelsize': 40,
#           'ytick.labelsize': 40,
#           'lines.linewidth': 6,
#           'text.usetex': True,
#           'figure.figsize': fig_size}

mp.rc('lines', lw=6)
mp.rc('savefig', format='pdf')
mp.rc('font', size = 40)
mp.rc('text', usetex = True)



def plot_afo_periodic():
    mp.rc('lines', lw=4)

    K = 100000.
    omegaF = 100.
    lamb = 1.
    dt = 0.000001
    save_dt = 0.0001
    t_end = 8
    omega0 = 20/lamb
    phi0 = 0
    
    #initialize an AFO object
    afo_obj = afos.PhaseAFO(K,omegaF,lamb)
    res = afos.integrate_afo(afo_obj,0,t_end,np.array([phi0,omega0]),dt,save_dt)
    
    t = res[0,:]
    phi = res[1,:]
    omega = res[2,:]

    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    ax1.plot(t, phi)
    plt.xlim([0, t_end])
    
    omega_bar_p = pi / (1 - exp(-lamb*pi/omegaF))
    omega_bar_m = pi / (exp(lamb*pi/omegaF) - 1)
    omega_bar_avg = 0.5*(omega_bar_p + omega_bar_m)
    
    ax2 = fig.add_subplot(212)
    ax2.plot(t, lamb*omega)
    ax2.plot(t, omegaF + ones(size(t))*lamb*pi, t, omegaF - ones(size(t))*lamb*pi, ls='--',color='k')
    ax2.plot(t, omega_bar_p * ones(size(t)), t, omega_bar_m * ones(size(t)), t, omega_bar_avg * ones(size(t)),ls=':')
    plt.xlim([0, t_end])
    

 
 
def plot_invariant_manifold_3D():
    mp.rc('font', size = 40)
    mp.rc('lines', lw=6)

    epsilon = 0.0001
    
    lambd = 1
    theta_m = arange(-1.0, -0.01, 0.01)
    theta_p = arange(0.01, 1.0, 0.01)
    Om = arange(-9.9,9.9,0.05)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    k_range = arange(0,10,1) 
    
    def invariant_k(k,ax):
        theta1p, Om1p = meshgrid(theta_p,Om)
        theta1m, Om1m = meshgrid(theta_m,Om)
        omega_p = (k*pi-Om1p)*(1+epsilon*((-1)**k)*lambd/theta1p) 
        omega_m = (k*pi-Om1m)*(1+epsilon*((-1)**k)*lambd/theta1m)
        if k%2==0: ##if k is even then F(theta)>0 for stable manifolds
#             print ((-1)**(k+1)*cos(omegaF*theta1p)).min()
#             print ((-1)**(k+1)*cos(omegaF*theta1p)).max()
            ax.plot_surface(theta1p, Om1p, omega_p, color='b', edgecolors='b') #stable
            ax.plot_surface(theta1m, Om1m, omega_m, color='r', edgecolors='r') #unstable
        else:
            ax.plot_surface(theta1p, Om1p, omega_p, color='r', edgecolors='r') #unstable
            ax.plot_surface(theta1m, Om1m, omega_m, color='b', edgecolors='b') #stable
       
    
    for k_var in k_range:
        invariant_k(k_var,ax)

    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(Om[0], Om[-1])
#     ax.set_zlim3d(-19, 5);
    
    
    #put the text    
    ax.text(1.35, 10.9, 16.0, r'$\displaystyle \omega$', zdir=None, fontsize=80)
    ax.text(-0.3, -12, -27.0, r'$\displaystyle F(\theta)$', zdir=None, fontsize=80)
    ax.text(1.4, 1.2, -16.0, r'$\displaystyle \Omega$', zdir=None, fontsize=80)
    

    #tune the view
    ax.azim = -74
    ax.elev = 28

    m = plt.get_current_fig_manager()
    m.resize(1591, 1273)

    fig.set_size_inches([ 19.8875,  15.9125])   
        

def plot_invariant_manifold_2D():
    mp.rc('lines', lw=6)

    epsilon = 0.0001
    
    lambd = 1
    theta_m = arange(-0.3, -0.0005, 0.001)
    theta_p = arange(0.001, 0.3, 0.001)
    Om = 1.0 #arange(-9.9,9.9,0.05)
    
    mp.rc('font', size=60)
    mp.rc('ytick', labelsize = 60)
    mp.rc('xtick', labelsize = 60)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    k_range = arange(-2,4,1) 
      
    
    def invariant_k(k,ax):
        omega_p = (k*pi-Om)*(1+epsilon*((-1)**k)*lambd/theta_p) 
        omega_m = (k*pi-Om)*(1+epsilon*((-1)**k)*lambd/theta_m)
        if k%2==0: ##if k is even then F(theta)>0 for stable manifolds
#             print ((-1)**(k+1)*cos(omegaF*theta1p)).min()
#             print ((-1)**(k+1)*cos(omegaF*theta1p)).max()
            ax.plot(theta_p, omega_p, color='b', lw = 6) #stable
            ax.plot(theta_m, omega_m, color='r', lw = 6,ls='--') #unstable
        else:
            ax.plot(theta_p, omega_p, color='r', lw = 6,ls='--') #unstable
            ax.plot(theta_m, omega_m, color='b', lw = 6) #stable
       
        ax.text(0.2, omega_p[-1]+0.5,'k={0}'.format(k))
    
    for k_var in k_range:
        invariant_k(k_var,ax)

    plt.xlim(-0.3,0.3)
    plt.ylim(-9.9,10)
        
    #put the text    
    plt.ylabel(r'$\displaystyle \omega$', fontsize=100)
    plt.xlabel(r'$\displaystyle F(\theta)$', fontsize=100)
        
    
    #tune the view
    m = plt.get_current_fig_manager()
    m.resize(1591, 1273)

    plt.subplots_adjust(left=0.12, bottom=0.18, top=0.97, right=0.92)

    fig.set_size_inches([ 19.8875,  15.9125])
 
