from figure_basics import *

# basic definitions
epsilon = 0.0001
lambd = 1

### plot the invariant manifolds in 2D
theta_m = arange(-0.3, -0.0005, 0.001)
theta_p = arange(0.001, 0.3, 0.001)
Om = 1.0 #arange(-9.9,9.9,0.05)

mp.rc('font', size=20)
mp.rc('ytick', labelsize = 20)
mp.rc('xtick', labelsize = 20)

fig = plt.figure(figsize=(7.5, 5.))
ax = fig.add_subplot(1,1,1)

k_range = arange(-2.,4,1) 


def invariant_k(k,ax):
    omega_p = (k*pi-Om)*(1+epsilon*((-1.)**k)*lambd/theta_p) 
    omega_m = (k*pi-Om)*(1+epsilon*((-1.)**k)*lambd/theta_m)
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
plt.ylabel(r'$\displaystyle \omega$', fontsize=30)
plt.xlabel(r'$\displaystyle F(\theta)$', fontsize=30)


#tune the view
m = plt.get_current_fig_manager()
m.resize(1591, 1273)

plt.subplots_adjust(left=0.12, bottom=0.18, top=0.97, right=0.92)



######################################
### plot the invariant manifolds in 3D
theta_m = arange(-1.0, -0.01, 0.01)
theta_p = arange(0.01, 1.0, 0.01)
Om = arange(-9.9,9.9,0.05)

fig = plt.figure(figsize=(9.5, 7.5))
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

#put the text
ax.text(1.35, 10.9, 16.0, r'$\displaystyle \omega$', zdir=None, fontsize=40)
ax.text(-0.3, -12, -27.0, r'$\displaystyle F(\theta)$', zdir=None, fontsize=40)
ax.text(1.4, 1.2, -16.0, r'$\displaystyle \Omega$', zdir=None, fontsize=40)

plt.xticks([-1,-0.5,0.,0.5,1.])
plt.yticks([-5,0,5])

#tune the 3D view
ax.azim = -74
ax.elev = 28

m = plt.get_current_fig_manager()
m.resize(1591, 1273)

plt.show()