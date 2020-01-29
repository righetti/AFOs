import numpy as np

## some functions useful for analysis

#cotangent hyperbolic function
def coth(z):
    return (np.exp(z)+np.exp(-z))/(np.exp(z)-np.exp(-z))

#returns f = sum of amp*cos(omega*t+phase)
def periodic_f(t,freq,amp,phase):

    f = np.zeros(np.size(t))
    for i in range(np.size(freq)):
        f = f + np.sin(freq[i]*t + phase[i]) * amp[i]
        
    return f

##returns the points of the +- omega maps for each zero crossing in times
##returns the +map and the -map
def compute_maps(times, omega_p0, omega_m0, lamb):
    omega_p = np.zeros(np.size(times))
    omega_m = np.zeros(np.size(times))
    omega_p[0] = omega_p0
    omega_m[0] = omega_m0
    delta_t = np.diff(times)
    for idx, dt in enumerate(delta_t):
        omega_p[idx+1] = omega_p[idx] * np.exp(-lamb*dt) + np.pi
        omega_m[idx+1] = (np.pi + omega_m[idx]) * np.exp(-lamb*dt)
        
    return omega_p, omega_m

#returns the zeros of a periodic function
#the zeros within the period
#we assume freq[0] is the base frequency
def find_roots(t,freq,amp,phase):
    omegaF = freq[0]
    
    f = periodic_f(t, freq, amp, phase)
    ind = []
    for i in range(np.size(f)-1): 
        if(f[i]*f[i+1]<0):
            ind.append(i)
    return t[ind]