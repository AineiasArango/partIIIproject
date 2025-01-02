import numpy as np
import matplotlib.pyplot as plt
import cosmo_utils as cu
import os
import read_fof_files as rff
os.chdir("/home/aasnha2/Project/Plots/mass_inflow_plots")

# Load the data
#get the virial radius of the particular snapshot
snap_dir2 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
snap_number = 86
fof_file = rff.get_fof_filename(snap_dir2, snap_number)
a = rff.get_attribute(fof_file, "Time") #scale factor
h = 0.679 #dimensionless Hubble constant
r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)*1000 #virial radius (pc)    

plt.rcParams.update({'font.size': 22})
plt.rcParams['axes.linewidth'] = 1.25
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.25

data = np.load('mass_inflow_rate_data.npz')
rads = data['rads']*1000 #convert to pc
mdot_ins1 = data['mdot_ins1']
mdot_ins2 = data['mdot_ins2']
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(rads, mdot_ins1, 'b-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(rads, mdot_ins2, linestyle='-', color='lime', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.axvline(r_vir, color='k', linestyle='--', linewidth=2)
ax1.set_xlabel('r [pc]')
ax1.set_xscale('log')
ax1.set_xticks([1e3, 1e4, r_vir], labels=['$10^3$', '$10^4$', 'R$_{\mathrm{vir}}$'])
ax1.set_xlim(3e2, rads[-1])
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='in')
ax1.tick_params(left=True, right=True)
ax1.set_ylabel('$\dot{\mathrm{M}}_{\mathrm{in}}(\mathrm{r})$ [M$_\odot \mathrm{yr}^{-1}$]')
ax1.set_yscale('log')
ax1.legend(frameon = False, fontsize=14, loc='lower left')
plt.savefig('mdot_in_vs_r.png')
plt.close()

data = np.load('mass_inflow_rate_T_data.npz')
rads = data['rads']*1000 #convert to pc
mdot_ins1_hot = data['mdot_ins1_hot']
mdot_ins1_cold = data['mdot_ins1_cold']
mdot_ins2_hot = data['mdot_ins2_hot']
mdot_ins2_cold = data['mdot_ins2_cold']
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(rads, mdot_ins1_hot, 'r-', linewidth=3, label='NoAGNHighSN (hot)') #NoBHFableHighSNEff
ax1.plot(rads, mdot_ins2_hot, linestyle='-', color='orange', linewidth=3, label='NoAGNHighSNRes (hot)') #NoBHFableHighSNEffHighRes
ax1.plot(rads, mdot_ins1_cold, 'b-', linewidth=3, label='NoAGNHighSN (cold)') #NoBHFableHighSNEff
ax1.plot(rads, mdot_ins2_cold, linestyle='-', color='aqua', linewidth=3, label='NoAGNHighSNRes (cold)') #NoBHFableHighSNEffHighRes
ax1.axvline(r_vir, color='k', linestyle='--', linewidth=2)
ax1.set_xlabel('r [pc]')
ax1.set_xscale('log')
ax1.set_xticks([1e3, 1e4, r_vir], labels=['$10^3$', '$10^4$', 'R$_{\mathrm{vir}}$'])
ax1.set_xlim(3e2, rads[-1])
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='out')
ax1.tick_params(left=True, right=True)
ax1.set_ylabel('$\dot{\mathrm{M}}_{\mathrm{in}}(\mathrm{r})$ [M$_\odot \mathrm{yr}^{-1}$]')
plt.yscale('log')
ax1.set_ylim(1e-6, 1e2)
ax1.legend(frameon = False, fontsize=11.5, loc='upper left', ncol=2)
plt.savefig('mdot_in_vs_r_T.png')
plt.close()

#Using just the data from the highres simulation, plot inflow and outflow
indata = np.load('mass_inflow_rate_data.npz')
outdata = np.load('mass_outflow_rate_data.npz')
rads = indata['rads']*1000 #convert to pc
mdot_ins = indata['mdot_ins2']
mdot_outs = outdata['mdot_outs2']
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(rads, mdot_ins, 'b-', linewidth=3, label='Inflow') #NoBHFableHighSNEff
ax1.plot(rads, mdot_outs, linestyle='-', color='lime', linewidth=3, label='Outflow') #NoBHFableHighSNEffHighRes
ax1.axvline(r_vir, color='k', linestyle='--', linewidth=2)
ax1.set_xlabel('r [pc]')
ax1.set_xscale('log')
ax1.set_xticks([1e3, 1e4, r_vir], labels=['$10^3$', '$10^4$', 'R$_{\mathrm{vir}}$'])
ax1.set_xlim(3e2, rads[-1])
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='in')
ax1.tick_params(left=True, right=True)
ax1.set_ylabel('$\dot{\mathrm{M}}(\mathrm{r})$ [M$_\odot \mathrm{yr}^{-1}$]')
ax1.set_yscale('log')
ax1.set_ylim(1e-6, 1e2)
ax1.legend(frameon = False, fontsize=14, loc='lower left')
plt.savefig('mdot_inout_vs_r.png')
plt.close()

indata = np.load('mass_inflow_rate_T_data.npz')
outdata = np.load('mass_outflow_rate_T_data.npz')
rads = indata['rads']*1000 #convert to pc
mdot_ins_hot = indata['mdot_ins2_hot']
mdot_outs_hot = outdata['mdot_outs2_hot']
mdot_ins_cold = indata['mdot_ins2_cold']
mdot_outs_cold = outdata['mdot_outs2_cold']
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(rads, mdot_ins_hot, 'r-', linewidth=3, label='Inflow (hot)') #NoBHFableHighSNEff
ax1.plot(rads, mdot_outs_hot, linestyle='-', color='orange', linewidth=3, label='Outflow (hot)') #NoBHFableHighSNEffHighRes
ax1.plot(rads, mdot_ins_cold, 'b-', linewidth=3, label='Inflow (cold)') #NoBHFableHighSNEff
ax1.plot(rads, mdot_outs_cold, linestyle='-', color='aqua', linewidth=3, label='Outflow (cold)') #NoBHFableHighSNEffHighRes
ax1.axvline(r_vir, color='k', linestyle='--', linewidth=2)
ax1.set_xlabel('r [pc]')
ax1.set_xscale('log')
ax1.set_xticks([1e3, 1e4, r_vir], labels=['$10^3$', '$10^4$', 'R$_{\mathrm{vir}}$'])
ax1.set_xlim(3e2, rads[-1])
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='out')
ax1.tick_params(left=True, right=True)
ax1.set_ylabel('$\dot{\mathrm{M}}(\mathrm{r})$ [M$_\odot \mathrm{yr}^{-1}$]')
plt.yscale('log')
ax1.set_ylim(1e-6, 1e2)
ax1.legend(frameon = False, fontsize=11.5, loc='upper left', ncol=2)
plt.savefig('mdot_inout_vs_r_T.png')
plt.close()