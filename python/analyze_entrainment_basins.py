import numpy as np
import pyafos

import matplotlib as mp
import matplotlib.pyplot as plt

import multiprocessing
import itertools
  

data = np.load('result2sin.npz')

avg_freq = data['avg_freq']
conv_freq = data['conv_freq']
KSweep = data['KSweep']
omegaSweep = data['omegaSweep']
omegaPert = data['omegaPert']
phasePert = data['phasePert']
amplitudePert = data['amplitudePert']
mean_error = data['mean_error']
max_error = data['max_error']
amp_error = data['amp_error']

# prepare the figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([omegaSweep[0], omegaSweep[-1]])
ax.set_ylim([KSweep[0], KSweep[-1]])
ax.set_xlabel(r'Frequency $[rad.s^{-1}]$')
ax.set_ylabel('Coupling K')
plt.yscale('log')

# do a colormap plot of the avg_freq
for om in omegaPert:
  print(om)
  temp_map = np.copy(avg_freq)
  temp_map[np.nonzero(np.abs(temp_map-om)<0.1)] = om
  temp_map[np.nonzero(np.abs(temp_map-om)>=0.1)] = om + 100
  temp_map = temp_map - om
  plt.contour(omegaSweep, KSweep, temp_map, 0, colors='k', linewidths=4, linestyles=':')
  #plt.pcolormesh(omegaSweep, KSweep, temp_map)

# do a contour plot of the convergence adaptive frequency
# conv_freq = np.floor(conv_freq)
plt.contour(omegaSweep, KSweep, conv_freq, 2, colors='k', linewidths=4)

# detect exponential convergence
# mean_error[np.nonzero(np.abs(mean_error)<0.01)] = 0.
# mean_error[np.nonzero(np.abs(mean_error)>=0.01)] = -.1
# plt.pcolormesh(omegaSweep, KSweep, mean_error, cmap='Greys', vmin=-0.1, vmax=0.2)
max_error = max_error/amp_error
threshold = 15.
max_error[np.nonzero(np.abs(max_error)<threshold)] = 0.
max_error[np.nonzero(np.abs(max_error)>=threshold)] = -.1
plt.pcolormesh(omegaSweep, KSweep, max_error, cmap='Greys', vmin=-.1, vmax = 0.1)
# plt.contour(omegaSweep, KSweep, max_error, 3, colors='k', linewidths=4, linestyles='--')

plt.show()

  

