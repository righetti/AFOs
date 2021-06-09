from figure_basics import *

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

mp.rc('lines', lw=12)
mp.rc('savefig', format='pdf')
mp.rc('font', size = 60)
mp.rc('text', usetex = True)



savename = 'resultFreqResp.npz'

data = np.load(savename)

amplitude = data['amplitude']
phase = data['phase']

omegaC_list = data["omegaC_list"]

lamb_list = data["lamb_list"]

plt.close('all')
    
fig = plt.figure(figsize=(19.8875,  15.9125))
m = plt.get_current_fig_manager()
m.resize(1591, 1273)
#axes 1
ax = fig.add_subplot(111)

plt.semilogx(omegaC_list, 20*np.log10(amplitude[0,:]))
plt.semilogx(omegaC_list, 20*np.log10(amplitude[1,:]))
plt.semilogx(omegaC_list, 20*np.log10(amplitude[2,:]))
plt.semilogx(omegaC_list, 0.0*np.ones_like(omegaC_list), 'k--', lw=6)
plt.semilogx(omegaC_list, -3.*np.ones_like(omegaC_list), 'k--', lw=6)
plt.semilogx(omegaC_list, -10.*np.ones_like(omegaC_list), 'k--', lw=6)
plt.semilogx(omegaC_list, -20.*np.ones_like(omegaC_list), 'k--', lw=6)
# plt.semilogx(omegaC_list, -30.*np.ones_like(omegaC_list), 'k--', lw=4)
# plt.semilogx(omegaC_list, -40.*np.ones_like(omegaC_list), 'k--', lw=4)
plt.semilogx([1,1],[2,-35], 'k--', lw=6)
plt.semilogx([.1,.1],[2,-35], 'k--', lw=6)
plt.semilogx([10,10],[2,-35], 'k--', lw=6)
#plt.semilogx([100,100],[2,-30], 'k--')
plt.ylim([-25,2])
plt.xlim([0.01, 100])
plt.yticks([-20,-10,-3,0])
ax.set_xlabel(r'$\displaystyle \omega_C$', size=80)
ax.set_ylabel(r'Amplitude [dB]', size=80)
fig.savefig("freqResp.pdf", bbox_inches='tight')

fig = plt.figure(figsize=(19.8875,  15.9125))
# fig = plt.figure()
m = plt.get_current_fig_manager()
m.resize(1591, 1273)
plt.semilogx(omegaC_list, phase[0,:])
plt.semilogx(omegaC_list, phase[1,:])
plt.semilogx(omegaC_list, phase[2,:])
plt.semilogx([1,1],[2,-35], 'k--', 'k--', lw=6)
plt.semilogx([.1,.1],[2,-35], 'k--', 'k--', lw=6)
plt.semilogx([10,10],[2,-35], 'k--', 'k--', lw=6)
plt.semilogx([100,100],[2,-35], 'k--', 'k--', lw=6)
plt.semilogx(omegaC_list, -np.pi/4.*np.ones_like(omegaC_list), 'k--', 'k--', lw=6)
plt.semilogx(omegaC_list, -np.pi/2.*np.ones_like(omegaC_list), 'k--', 'k--', lw=6)
plt.semilogx(omegaC_list, -3*np.pi/4.*np.ones_like(omegaC_list), 'k--', 'k--', lw=6)
plt.semilogx(omegaC_list, -np.pi*np.ones_like(omegaC_list), 'k--', 'k--', lw=6)
plt.semilogx(omegaC_list, -2*np.pi*np.ones_like(omegaC_list), 'k--', 'k--', lw=6)
plt.xlim([0.01, 100])
plt.ylim([-np.pi/2.-0.1,0])
plt.xlabel(r'$\displaystyle \omega_C$', size=80)
plt.ylabel(r'Phase [rad]', size=80)
plt.yticks([-np.pi/2, -np.pi/4, 0],[r'$-\frac{\pi}{2}$',r'$-\frac{\pi}{4}$',r'$0$'])
fig.savefig("freqRespAngle.pdf", bbox_inches='tight')

plt.show()