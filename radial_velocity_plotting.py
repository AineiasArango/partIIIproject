import numpy as np
import matplotlib.pyplot as plt
import cosmo_utils as cu
import os
from scipy import constants
import read_fof_files as rff

Msun = 1.988416e30 #kg

#get the virial radius of the particular snapshot
snap_dir1 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
snap_number = 86
fof_file = rff.get_fof_filename(snap_dir1, snap_number)
a = rff.get_attribute(fof_file, "Time") #scale factor
h = 0.679 #dimensionless Hubble constant
r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)*1000 #virial radius (pc)  

#Calculate Keplerian velocity
os.chdir("/home/aasnha2/Project/Plots/mass_plots")
massdata = np.load('total_mass_data.npz')
rads = massdata['rads']*1000 #convert to pc
masses1 = massdata['total_masses1']
masses2 = massdata['total_masses2']
unit_conversion = Msun/3.0857e16
v_ks1 = np.sqrt(constants.G*masses1*unit_conversion/rads)/1000
v_ks2 = np.sqrt(constants.G*masses2*unit_conversion/rads)/1000
os.chdir("/home/aasnha2/Project/Plots/radial_velocity_plots")

plt.rcParams.update({'font.size': 22})
plt.rcParams['axes.linewidth'] = 1.25
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.25

data = np.load('radial_velocity_data.npz')
v_radials1 = data['v_radials1']
v_radials2 = data['v_radials2']
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(rads, v_radials1, 'b-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, v_radials2, linestyle='-', color='lime', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.plot(rads, v_ks2, linestyle='-', color='dimgrey', linewidth=2, label='v$_{\mathrm{K}}$')
ax1.axvline(r_vir, color='k', linestyle='--', linewidth=2)
ax1.set_xlabel('r [pc]')
ax1.set_xscale('log')
ax1.set_xticks([1e3, 1e4, r_vir], labels=['$10^3$', '$10^4$', 'R$_{\mathrm{vir}}$'])
ax1.set_xlim(3e2, rads[-1])
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='out')
ax1.tick_params(left=True, right=True)
ax1.set_ylabel('v$_{\mathrm{r}}$ [km/s]')
ax1.set_yscale('log')
ax1.legend(frameon = False, fontsize=14, loc='lower left')
plt.savefig('v_radial_vs_r.png')
plt.close()

data = np.load('radial_velocity_T_data.npz')
rads = data['rads']*1000 #convert to pc
v_radials_hot1 = data['v_radials_hot1']
v_radials_hot2 = data['v_radials_hot2']
v_radials_cold1 = data['v_radials_cold1']
v_radials_cold2 = data['v_radials_cold2']
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(rads, v_radials_hot1, 'r-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, v_radials_hot2, linestyle='-', color='orange', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.plot(rads, v_radials_cold1, 'b-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, v_radials_cold2, linestyle='-', color='aqua', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.plot(rads, v_ks2, linestyle='-', color='dimgrey', linewidth=2, label='v$_{\mathrm{K}}$')
ax1.axvline(r_vir, color='k', linestyle='--', linewidth=2)
ax1.set_xlabel('r [pc]')
ax1.set_xscale('log')
ax1.set_xticks([1e3, 1e4, r_vir], labels=['$10^3$', '$10^4$', 'R$_{\mathrm{vir}}$'])
ax1.set_xlim(3e2, rads[-1])
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='out')
ax1.tick_params(left=True, right=True)
ax1.set_ylabel('v$_{\mathrm{r}}$ [km/s]')
ax1.set_yscale('log')
ax1.legend(bbox_to_anchor=(0.5, 1.2),fontsize=13, loc='upper center', ncol=2)
plt.savefig('v_radial_vs_r_T.png')
plt.close()
