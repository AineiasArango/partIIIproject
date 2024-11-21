import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import constants

os.chdir("/home/aasnha2/Project/Plots")

Msun = 1.988416e30 #kg

# Load the data
data = np.load('radial_velocity_T5_data.npz')
rads = data['rads']*1000 #convert to pc
v_radials_hot1 = data['v_radials_hot1']
v_radials_hot2 = data['v_radials_hot2']
v_radials_cold1 = data['v_radials_cold1']
v_radials_cold2 = data['v_radials_cold2']

#Calculate Keplerian velocity
massdata = np.load('mass_data.npz')
masses1 = massdata['masses1']
masses2 = massdata['masses2']
unit_conversion = Msun/3.0857e16
v_ks1 = np.sqrt(constants.G*masses1*unit_conversion/rads)/1000
v_ks2 = np.sqrt(constants.G*masses2*unit_conversion/rads)/1000

plt.rcParams.update({'font.size': 22})
plt.rcParams['axes.linewidth'] = 1.25
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.25

fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(rads, v_radials_hot1, 'r-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, v_radials_hot2, linestyle='-', color='orange', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.plot(rads, v_radials_cold1, 'b-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, v_radials_cold2, linestyle='-', color='aqua', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.plot(rads, v_ks2, linestyle='-', color='dimgrey', linewidth=2, label='v$_{\mathrm{K}}$')
ax1.set_xlabel('r [pc]')
ax1.set_xscale('log')
ax1.set_xticks([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
ax1.set_xlim(3e2, rads[-1])
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='out')
ax1.tick_params(left=True, right=True)
ax1.set_ylabel('v$_{\mathrm{r}}$ [km/s]')
ax1.set_ylim(0, 26)
#ax1.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.1], minor=True)
ax1.legend(frameon = False, fontsize=13, loc='upper right')
plt.savefig('v_radial_vs_r_T5.png')
plt.close()
