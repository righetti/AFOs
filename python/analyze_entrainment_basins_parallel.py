import numpy as np
import pyafos

import matplotlib as mp
import matplotlib.pyplot as plt

import multiprocessing
import itertools
import ctypes

if __name__ == '__main__':

  # the perturbation
  omegaPert = np.array([30,60,90])
  amplitudePert = np.array([1.3,1,1.4])
  phasePert = np.array([0.4,0,1.3])

  savename = 'result2sin.npz'
 
  KSweep = np.logspace(1,3,200)
  omegaSweep = np.linspace(20., 100., 200)
 
  dt = 0.0001
  save_dt = 0.001
  lamb = 1.
 
  numK = np.size(KSweep)
  numOm = np.size(omegaSweep)

  # we create shared memory (needed to change the same state of variables)
  # we do not need to do shared memory for read-only variables because Posix
  # ensures that the fork copies the state of all variables (might not work on Windows)
  avg_freq_shared = multiprocessing.RawArray(ctypes.c_double, numK*numOm)
  conv_freq = multiprocessing.RawArray(ctypes.c_double, numK*numOm)
  max_error = multiprocessing.RawArray(ctypes.c_double, numK*numOm)
  mean_error = multiprocessing.RawArray(ctypes.c_double, numK*numOm)
  amp_error = multiprocessing.RawArray(ctypes.c_double, numK*numOm)

  avg_freq = np.frombuffer(avg_freq_shared).reshape((numK, numOm))
  conv_freq = np.frombuffer(conv_freq).reshape((numK, numOm))
  max_error = np.frombuffer(max_error).reshape((numK, numOm))
  mean_error = np.frombuffer(mean_error).reshape((numK, numOm))
  amp_error = np.frombuffer(amp_error).reshape((numK, numOm))

  process_mutex = multiprocessing.Lock()

  def process_parallel(i,j):
    K = KSweep[i]
    om = omegaSweep[j]

    # if K < 10:
    #   T = 300.
    # elif K < 50:
    #   T = 200.
    # else:
    #   T = 100.
    T = 300.

    print("processing K = " + str(K) + " and omega = " + str(om))

    oscill = pyafos.DualPhasePhaseAFO()
    oscill.input().vec_of_sines(omegaPert, amplitudePert, phasePert)

    oscill.initialize(K, lamb, om)
    pyafos.integrate(oscill, 0.0, T, np.array([0.,0.,om]), dt, save_dt)

    t = oscill.t()
    mid_point = int(3*len(t)/4)
    phiNormal = oscill.y()[0,:]
    omegaAFO = oscill.y()[2,:]
    
    # the average frequency of oscillations of the normal phase osc.
    avg_freq[i,j] = np.mean(np.diff(phiNormal)/save_dt)

    # the frequency of convergence of omega
    conv_freq[i,j] = np.mean(omegaAFO[mid_point:])
    
    # the max error between the mean converged frequency and the oscillations
    amp_error[i,j] = np.max(np.abs(omegaAFO[mid_point:]-conv_freq[i,j]))
    
    # compute the frequency toward which it has converged (from the input signal)
    dis = np.abs(omegaPert-conv_freq[i,j])
    f = omegaPert[np.argwhere( dis == np.min(dis))][0,0]
    
    # print('converged to: ' + str(f) + ' with conv freq ' + str(conv_freq[i,j]) + 'normal oscill avg. freq ' + str(avg_freq[i,j]))
    
    # if the conv_freq is close enough to f - then compute
    # mean error between exponential convergence using F and omega
    # max error between exponential convergence using F and omega
    if np.abs(f-conv_freq[i,j])>2*np.pi:
      mean_error[i,j] = float('nan')
      max_error[i,j] = float('nan')
      print('NaN')
    else:
      y_avg = f + (om-f)*np.exp(-t)
      mean_error[i,j] = np.mean(omegaAFO-y_avg)
      max_error[i,j] = np.max(np.abs(omegaAFO-y_avg))


  pool = multiprocessing.Pool()
  pool.starmap(process_parallel, list(itertools.product(list(range(len(KSweep))), list(range(len(omegaSweep))))))
  pool.close()

  np.savez(savename, avg_freq=avg_freq, conv_freq=conv_freq, KSweep=KSweep, omegaSweep=omegaSweep, omegaPert=omegaPert, phasePert=phasePert, amplitudePert=amplitudePert, mean_error=mean_error, max_error=max_error, amp_error=amp_error)
