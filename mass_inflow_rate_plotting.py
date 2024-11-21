import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
import scipy
import cosmo_utils as cu
from scipy import stats
import os
import matplotlib.ticker as tck
os.chdir("/home/aasnha2/Project/Plots")

# Load the data
data = np.load('mass_inflow_rate_data2119.npz')
rads = data['rads']*1000 #convert to pc
mdot_ins1 = data['mdot_ins1']
mdot_ins2 = data['mdot_ins2']

plt.rcParams.update({'font.size': 22})
plt.rcParams['axes.linewidth'] = 1.25
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.25

fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(rads, mdot_ins1, 'b-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, mdot_ins2, linestyle='-', color='lime', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.set_xlabel('r [pc]')
ax1.set_xscale('log')
ax1.set_xticks([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
ax1.set_xlim(3e2, rads[-1])
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='out')
ax1.tick_params(left=True, right=True)
ax1.set_ylabel('$\dot{\mathrm{M}}_{\mathrm{in}}(\mathrm{r})$ [M$_\odot \mathrm{yr}^{-1}$]')
ax1.set_ylim(0, 1.1)
ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
ax1.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.1], minor=True)
ax1.legend(frameon = False, fontsize=14, loc='upper left')
plt.savefig('mdot_in_vs_r.png')
plt.close()