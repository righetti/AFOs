import pyafos

import scipy.signal
import numpy as np
import matplotlib.pylab as plt

import itertools

from concurrent.futures import ProcessPoolExecutor

def find_envelop(data):
    maximums = []
    minimums = []
    x_max = data[0]
    seek_max = True
    for i,x in enumerate(data):
        if seek_max:
            if x_max <= x:
                x_max = x
            else:
                maximums.append(np.array([i-1,x_max]))
                x_max = x
                seek_max = False
        else:
            if x_max >= x:
                x_max = x
            else:
                minimums.append(np.array([i-1,x_max]))
                x_max = x
                seek_max = True
    return np.array(minimums), np.array(maximums)


def find_frequency_response(arg):
    lamb = arg[0]
    omegaC = arg[1]

    K = 10.**5
    omegaF = 1000.

    dt = 10**-5
    
    if omegaC < 50.:
        save_dt = 10**-4
    else:
        save_dt = 10**-5



    # here the convergence rate is too slow - leads to large transcient
    if lamb < 1.:
        min_time = 200.
    else:
        min_time = 100.

    max_time = 10000.

    
    print("processing lambda = " + str(lamb) + " and omegaC = " + str(omegaC))

    oscill = pyafos.PhaseAFO()
    oscill.initialize(K, lamb)
    oscill.input().frequency_changing_sine(omegaF, omegaC)

    # set initial conditions
    # we want to measure at least 50 oscillations of omegaC
    # 50 * 2pi/omegaC
    t_end = np.min([np.max([50 * 2 * np.pi / omegaC, min_time]),max_time])
    t_start = 0.
    omega0 = (omegaF + 1.) / lamb
    phi0 = 0.


    pyafos.integrate(oscill, t_start,t_end,np.array([phi0,omega0]),dt,save_dt)

    omega = lamb * oscill.y()[1,:]
    t = oscill.t()

    mins, maxs = find_envelop(omega)

    n_min = mins[:,1].size
    n_max = maxs[:,1].size
    if n_min < n_max:
        track = 0.5*(mins[:,1]+maxs[:-(n_max-n_min),1])
        track_signal = omegaF + np.cos(omegaC*t[mins.astype(int)[:,0]])
    elif n_min == n_max:
        track = 0.5*(mins[:,1]+maxs[:,1])
        track_signal = omegaF + np.cos(omegaC*t[mins.astype(int)[:,0]])
    else:
        track = 0.5*(mins[:,1]+maxs[:-(n_min-n_max),1])
        track_signal = omegaF + np.cos(omegaC*t[maxs.astype(int)[:,0]])
    
    # we get rid of the first half of the signal, make sure no transcients are here
    track = track[int(track.size/2):]
    track_signal = track_signal[int(track_signal.size/2):]

    response = scipy.signal.hilbert(track-np.mean(track)) / scipy.signal.hilbert(track_signal-omegaF)

    # we only keep the middle third of the response to remove boarder effects
    si = response.size
    am = np.mean(np.abs(response[int(si/3):int(2*si/3)]))
    #to make sure all is close to each other
    angle = np.unwrap(np.angle(response[int(si/3):int(2*si/3)]))
    ph = np.mean(angle-2*np.pi)%(-2*np.pi)
    if np.abs(ph) > 1.8*np.pi:
        ph = ph+2*np.pi
    # ph = np.mean((np.angle(response[int(si/3):int(2*si/3)]) - 2 * np.pi))%(-2*np.pi)
    print("done with lambda = " + str(lamb) + " and omegaC = " + str(omegaC) + "found " + str(am) + " " + str(ph))
    
    # amplitude[i,j] = am
    # phase[i,j] = ph
    return am, ph

# here we run systematic evaluation of the frequency response of the system
if __name__ == '__main__':

    omegaC_list = np.logspace(-2, 2, 50)
    num_omegaC = omegaC_list.size

    lamb_list = np.logspace(-1,1,3)
    num_lamb = lamb_list.size


    amplitude = np.zeros([num_lamb, num_omegaC])
    phase = np.zeros([num_lamb, num_omegaC])

    data = np.load('resultFreqResp.npz')
    amplitude = data['amplitude']
    phase = data['phase']

    savename = 'resultFreqResp2.npz'
    
    with ProcessPoolExecutor() as executor:
        # tasks_iter = list(itertools.product(list(range(len(lamb_list))), list(range(len(omegaC_list)))))
        # params_iter = list(itertools.product(lamb_list, omegaC_list))
        # for t, res in zip(tasks_iter, executor.map(find_frequency_response, params_iter)):
        #     amplitude[t[0], t[1]] = res[0]
        #     phase[t[0], t[1]] = res[1]

        tasks_iter = list(itertools.product(list(range(len(lamb_list))), list(range(len(omegaC_list)))))
        params_iter = list(itertools.product(lamb_list, omegaC_list))
        for t, res in zip(tasks_iter, executor.map(find_frequency_response, params_iter)):
            amplitude[t[0],t[1]] = res[0]
            phase[t[0],t[1]] = res[1]

    np.savez(savename, 
            amplitude=amplitude, phase=phase, 
            omegaC_list=omegaC_list, lamb_list=lamb_list)








    