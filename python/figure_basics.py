import numpy as np
import matplotlib.pyplot as plt
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



# function to create a subax
def create_subax(fig, ax, rect, xlimit=[], ylimit=[], xticks=[], yticks=[], side='b', xticklab=[], yticklab=[]):
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    sub_ax = fig.add_axes([x,y,width,height])
    
    if xlimit:
        sub_ax.set_xlim(xlimit)
    if ylimit:
        sub_ax.set_ylim(ylimit)

    for tick in sub_ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(30)
    for tick in sub_ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(30)

    
    sub_ax.set_xticks(xticks)
    sub_ax.set_yticks(yticks)
    if xticklab:
        sub_ax.set_xticklabels(xticklab)
    if yticklab:
        sub_ax.set_yticklabels(yticklab)
    
    if xlimit and ylimit:
        rect1 = mp.patches.Rectangle((xlimit[0],ylimit[0]), xlimit[1]-xlimit[0], ylimit[1]-ylimit[0], 
                                    color='k', fill=False, lw=2, zorder=5)
        ax.add_patch(rect1)
        transData = ax.transData.inverted()
        if side == 'b':
            subax_pos1 = transData.transform(ax.transAxes.transform(np.array(rect[0:2])+np.array([0,rect[3]])))
            subax_pos2 = transData.transform(ax.transAxes.transform(np.array(rect[0:2])+np.array([rect[2],rect[3]]))) 
            ax.plot([xlimit[0],subax_pos1[0]],[ylimit[0],subax_pos1[1]], color='k', lw=2)
            ax.plot([xlimit[1],subax_pos2[0]],[ylimit[0],subax_pos2[1]], color='k', lw=2)
        elif side == 'r':
            subax_pos1 = transData.transform(ax.transAxes.transform(np.array(rect[0:2])+np.array([0,rect[3]])))
            subax_pos2 = transData.transform(ax.transAxes.transform(np.array(rect[0:2]))) 
            ax.plot([xlimit[1],subax_pos1[0]],[ylimit[1],subax_pos1[1]], color='k', lw=2)
            ax.plot([xlimit[1],subax_pos2[0]],[ylimit[0],subax_pos2[1]], color='k', lw=2)
        elif side == 't':
            subax_pos1 = transData.transform(ax.transAxes.transform(np.array(rect[0:2])))
            subax_pos2 = transData.transform(ax.transAxes.transform(np.array(rect[0:2])+np.array([rect[2],0]))) 
            ax.plot([xlimit[0],subax_pos1[0]],[ylimit[1],subax_pos1[1]], color='k', lw=2)
            ax.plot([xlimit[1],subax_pos2[0]],[ylimit[1],subax_pos2[1]], color='k', lw=2)
    
    return sub_ax