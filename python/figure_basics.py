import numpy as np
import matplotlib.pylab as plt
import importlib
import matplotlib as mp
from numpy import exp
from mpl_toolkits.mplot3d import axes3d
from numpy import sqrt, arange, pi, meshgrid, cos, sin, mod, size, ones, zeros, linspace, floor, exp

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