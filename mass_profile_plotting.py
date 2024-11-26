import numpy as np
import matplotlib.pyplot as plt
import cosmo_utils as cu
import os
import read_fof_files as rff
os.chdir("/home/aasnha2/Project/Plots/mass_plots")

#get the virial radius of the particular snapshot
snap_dir1 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
snap_number = 86
fof_file = rff.get_fof_filename(snap_dir1, snap_number)
a = rff.get_attribute(fof_file, "Time") #scale factor
h = 0.679 #dimensionless Hubble constant
r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)*1000 #virial radius (pc)    


plt.rcParams.update({'font.size': 22})
plt.rcParams['axes.linewidth'] = 1.25
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.25

#internal mass no temp split
data = np.load('mass_data.npz')
rads = data['rads']*1000 #convert to pc
masses1 = data['masses1']
masses2 = data['masses2']
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.set_title('Cumulative gas mass profile')
ax1.plot(rads, masses1, 'b-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, masses2, linestyle='-', color='lime', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.axvline(r_vir, color='k', linestyle='--', linewidth=2)
ax1.set_xlabel('r [pc]')
ax1.set_xscale('log')
ax1.set_xticks([1e3, 1e4, r_vir], labels=['$10^3$', '$10^4$', 'R$_{\mathrm{vir}}$'])
ax1.set_xlim(3e2, rads[-1])
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='out')
ax1.tick_params(left=True, right=True)
ax1.set_ylabel('M$_{\mathrm{gas}}(\mathrm{r})$ [M$_\odot$]')
ax1.set_ylim(1e4, 1e9)
ax1.set_yscale('log')
ax1.legend(frameon = False, fontsize=14, loc='lower left')
plt.savefig('internal_mass_profile.png')
plt.close()

#mass density
data = np.load('mass_density_data.npz')
rads = data['rads']*1000 #convert to pc
mass_densities1 = data['mass_densities1']
mass_densities2 = data['mass_densities2']
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.set_title('Gas mass profile')
ax1.plot(rads, mass_densities1, 'b-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, mass_densities2, linestyle='-', color='lime', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.axvline(r_vir, color='k', linestyle='--', linewidth=2)
ax1.set_xlabel('r [pc]')
ax1.set_xscale('log')
ax1.set_xticks([1e3, 1e4, r_vir], labels=['$10^3$', '$10^4$', 'R$_{\mathrm{vir}}$'])
ax1.set_xlim(3e2, rads[-1])
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='out')
ax1.tick_params(left=True, right=True)
ax1.set_ylabel('m$_{\mathrm{gas}}(\mathrm{r})$ [M$_\odot$/pc]')
#ax1.set_ylim(1e4, 1e9)
ax1.set_yscale('log')
ax1.legend(frameon = False, fontsize=14, loc='lower left')
plt.savefig('mass_density_profile.png')
plt.close()

#internal mass temp split
data = np.load('mass_data_temp_split.npz')
rads = data['rads']*1000 #convert to pc
masses_hot1 = data['masses_hot1']
masses_cold1 = data['masses_cold1']
masses_hot2 = data['masses_hot2']
masses_cold2 = data['masses_cold2']
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(rads, masses_hot1, 'r-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, masses_hot2, linestyle='-', color='orange', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.plot(rads, masses_cold1, 'b-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, masses_cold2, linestyle='-', color='aqua', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.axvline(r_vir, color='k', linestyle='--', linewidth=2)
ax1.set_xlabel('r [pc]')
ax1.set_xscale('log')
ax1.set_xticks([1e3, 1e4, r_vir], labels=['$10^3$', '$10^4$', 'R$_{\mathrm{vir}}$'])
ax1.set_xlim(3e2, rads[-1])
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='out')
ax1.tick_params(left=True, right=True)
ax1.set_ylabel('M$_{\mathrm{gas}}(\mathrm{r})$ [M$_\odot$]')
ax1.set_ylim(1e4, 1e9)
ax1.set_yscale('log')
ax1.legend(frameon = False, fontsize=13, loc='lower right')
plt.savefig('internal_mass_temp_split.png')
plt.close()

#mass density temp split
data = np.load('mass_density_data_temp_split.npz')
rads = data['rads']*1000 #convert to pc
mass_densities_hot1 = data['mass_densities_hot1']
mass_densities_cold1 = data['mass_densities_cold1']
mass_densities_hot2 = data['mass_densities_hot2']
mass_densities_cold2 = data['mass_densities_cold2']
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(rads, mass_densities_hot1, 'r-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, mass_densities_hot2, linestyle='-', color='orange', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.plot(rads, mass_densities_cold1, 'b-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, mass_densities_cold2, linestyle='-', color='aqua', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.axvline(r_vir, color='k', linestyle='--', linewidth=2)
ax1.set_xlabel('r [pc]')
ax1.set_xscale('log')
ax1.set_xticks([1e3, 1e4, r_vir], labels=['$10^3$', '$10^4$', 'R$_{\mathrm{vir}}$'])
ax1.set_xlim(3e2, rads[-1])
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='out')
ax1.tick_params(left=True, right=True)
ax1.set_ylabel('m$_{\mathrm{gas}}(\mathrm{r})$ [M$_\odot$/pc]')
ax1.set_ylim(1e4, 1e9)
ax1.set_yscale('log')
ax1.legend(frameon = False, fontsize=13, loc='upper right')
plt.savefig('mass_density_temp_split.png')
plt.close()

#total mass
data = np.load('total_mass_data.npz')
rads = data['rads']*1000 #convert to pc
total_masses1 = data['total_masses1']
total_masses2 = data['total_masses2']
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.set_title('Cumulative total mass profile')
ax1.plot(rads, total_masses1, 'b-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, total_masses2, linestyle='-', color='lime', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.axvline(r_vir, color='k', linestyle='--', linewidth=2)
ax1.set_xlabel('r [pc]')
ax1.set_xscale('log')
ax1.set_xticks([1e3, 1e4, r_vir], labels=['$10^3$', '$10^4$', 'R$_{\mathrm{vir}}$'])
ax1.set_xlim(3e2, rads[-1])
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='out')
ax1.tick_params(left=True, right=True)
ax1.set_ylabel('M$_{\mathrm{gas}}(\mathrm{r})$ [M$_\odot$]')
ax1.set_ylim(1e7, 1e10)
ax1.set_yscale('log')
ax1.legend(frameon = False, fontsize=14, loc='upper left')
plt.savefig('total_mass_profile.png')
plt.close()