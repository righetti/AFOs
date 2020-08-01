from figure_basics import *

mp.rc('lines', lw=4)
mp.rc('font', size=60)

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
plt.semilogx(omegaC_list, 0.0*np.ones_like(omegaC_list), 'k--', lw=4)
plt.semilogx(omegaC_list, -3.*np.ones_like(omegaC_list), 'k--', lw=4)
plt.semilogx(omegaC_list, -10.*np.ones_like(omegaC_list), 'k--', lw=4)
plt.semilogx(omegaC_list, -20.*np.ones_like(omegaC_list), 'k--', lw=4)
# plt.semilogx(omegaC_list, -30.*np.ones_like(omegaC_list), 'k--', lw=4)
# plt.semilogx(omegaC_list, -40.*np.ones_like(omegaC_list), 'k--', lw=4)
plt.semilogx([1,1],[2,-35], 'k--', lw=4)
plt.semilogx([.1,.1],[2,-35], 'k--', lw=4)
plt.semilogx([10,10],[2,-35], 'k--', lw=4)
#plt.semilogx([100,100],[2,-30], 'k--')
plt.ylim([-25,2])
plt.xlim([0.01, 100])
plt.yticks([-20,-3,0])
ax.set_xlabel(r'$\displaystyle \omega_C$', size=80)
ax.set_ylabel(r'Amplitude [dB]', size=80)

plt.show()
# plt.figure()
# plt.semilogx(omegaC_list, phase[0,:])
# plt.semilogx(omegaC_list, phase[1,:])
# plt.semilogx(omegaC_list, phase[2,:],'x')
# plt.semilogx([1,1],[2,-35], 'k--')
# plt.semilogx([.1,.1],[2,-35], 'k--')
# plt.semilogx([10,10],[2,-35], 'k--')
# plt.semilogx([100,100],[2,-35], 'k--')
# plt.semilogx(omegaC_list, -np.pi/4.*np.ones_like(omegaC_list), 'k--')
# plt.semilogx(omegaC_list, -np.pi/2.*np.ones_like(omegaC_list), 'k--')
# plt.semilogx(omegaC_list, -3*np.pi/4.*np.ones_like(omegaC_list), 'k--')
# plt.semilogx(omegaC_list, -np.pi*np.ones_like(omegaC_list), 'k--')
# plt.xlim([0.01, 100])
# plt.ylim([-np.pi,0])

