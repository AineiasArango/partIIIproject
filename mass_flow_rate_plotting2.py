import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15
import scipy
import cosmo_utils as cu
from scipy import stats
import os
os.chdir("/home/aasnha2/Project/Plots")

# Load the data
data = np.load('mass_flow_rate_data.npz')

# Extract the data
redshifts1 = data['redshifts1']
mdot_tot1 = data['mdot_tot1']
mdot_in1 = np.abs(data['mdot_in1'])
mdot_out1 = data['mdot_out1']
beta1 = data['beta1']
cosmic_time1 = Planck15.age(redshifts1).value

redshifts2 = data['redshifts2']
mdot_tot2 = data['mdot_tot2']
mdot_in2 = np.abs(data['mdot_in2'])
mdot_out2 = data['mdot_out2']
beta2 = data['beta2']
cosmic_time2 = Planck15.age(redshifts2).value

# Plotting parameters
plt.rcParams.update({'font.size': 22})
plt.rcParams['axes.linewidth'] = 1.25
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.25

# Plot mdot_in vs cosmic time
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(cosmic_time1, mdot_in1, 'b-', linewidth=3, label='NoBHFableHighSNEff')
ax1.plot(cosmic_time2, mdot_in2, linestyle='-', color='lime', linewidth=3, label='NoBHFableHighSNEffHighRes')
ax1.set_xlabel('t [Gyr]')
ax1.set_xticks([2, 3, 4, 5])
ax1.set_xlim(Planck15.age(4).value, Planck15.age(1).value)
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='in')
plt.gca().invert_xaxis()
ax1.set_ylabel('$\dot{\mathrm{M}}_{\mathrm{in}}(\mathrm{R}_{\mathrm{vir}})$ [M$_\odot \mathrm{yr}^{-1}$]')
ax1.set_yscale('log')
ax1.set_yticks([1e-3, 1e-2, 1e-1])
ax1.set_ylim(4e-4, 3e-1)
ax1.legend(frameon = False, fontsize=11)
cu.add_redshifts_ticks(fig, int(1), int(4), fontsize=22)
plt.savefig('mass_inflow_rate_vs_cosmic_time.png')
plt.close()

# Plot mdot_out vs cosmic time
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(cosmic_time1, mdot_out1, 'b-', linewidth=3, label='NoBHFableHighSNEff')
ax1.plot(cosmic_time2, mdot_out2, linestyle='-', color='lime', linewidth=3, label='NoBHFableHighSNEffHighRes')
ax1.set_xlabel('t [Gyr]')
ax1.set_xticks([2, 3, 4, 5])
ax1.set_xlim(Planck15.age(4).value, Planck15.age(1).value)
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='in')
plt.gca().invert_xaxis()
ax1.set_ylabel('$\dot{\mathrm{M}}_{\mathrm{out}}(\mathrm{R}_{\mathrm{vir}})$ [M$_\odot \mathrm{yr}^{-1}$]')
ax1.set_yscale('log')
ax1.set_yticks([1e-3, 1e-2, 1e-1])
ax1.set_ylim(4e-4, 3e-1)
ax1.legend(frameon = False, fontsize=11)
cu.add_redshifts_ticks(fig, int(1), int(4), fontsize=22)
plt.savefig('mass_outflow_rate_vs_cosmic_time.png')
plt.close()

# Plot beta vs cosmic time
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(cosmic_time1, beta1, 'b-', linewidth=3, label='NoBHFableHighSNEff')
ax1.plot(cosmic_time2, beta2, linestyle='-', color='lime', linewidth=3, label='NoBHFableHighSNEffHighRes')
ax1.set_xlabel('t [Gyr]')
ax1.set_xticks([2, 3, 4, 5])
ax1.set_xlim(Planck15.age(4).value, Planck15.age(1).value)
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='in')
plt.gca().invert_xaxis()
ax1.set_ylabel(r'$\mathrm{\beta}_{\mathrm{out}}(\mathrm{R}_{\mathrm{vir}})$')
ax1.set_yscale('log')
ax1.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
ax1.set_ylim(1e-2, 2e3)
ax1.legend(frameon = False, fontsize=11)
cu.add_redshifts_ticks(fig, int(1), int(4), fontsize=22)
plt.savefig('beta_vs_cosmic_time.png')
plt.close()


#Plot mdot_in vs cosmic time with binned data
bins1 = np.append(np.arange(Planck15.age(6).value, max(cosmic_time1), 0.5), max(cosmic_time1))
bins2 = np.append(np.arange(Planck15.age(6).value, max(cosmic_time2), 0.5), max(cosmic_time2))
binned_mdot_in1, bin_edges1, binnumber1 = stats.binned_statistic(cosmic_time1, mdot_in1, statistic='mean', bins=bins1)
binned_mdot_in2, bin_edges2, binnumber2 = stats.binned_statistic(cosmic_time2, mdot_in2, statistic='mean', bins=bins2)
binned_cosmic_time1 = np.append(bin_edges1[:-2] + 0.25, bin_edges1[-2] + (bin_edges1[-1]-bin_edges1[-2])/2)
binned_cosmic_time2 = np.append(bin_edges2[:-2] + 0.25, bin_edges2[-2] + (bin_edges2[-1]-bin_edges2[-2])/2)
binned_mdot_out1, bin_edges1, binnumber1 = stats.binned_statistic(cosmic_time1, mdot_out1, statistic='mean', bins=bins1)
binned_mdot_out2, bin_edges2, binnumber2 = stats.binned_statistic(cosmic_time2, mdot_out2, statistic='mean', bins=bins2)
binned_beta1, bin_edges1, binnumber1 = stats.binned_statistic(cosmic_time1, beta1, statistic='mean', bins=bins1)
binned_beta2, bin_edges2, binnumber2 = stats.binned_statistic(cosmic_time2, beta2, statistic='mean', bins=bins2)

fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(binned_cosmic_time1, binned_mdot_in1, 'b-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(binned_cosmic_time2, binned_mdot_in2, linestyle='-', color='lime', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.set_xlabel('t [Gyr]')
ax1.set_xticks([2, 3, 4, 5])
ax1.set_xlim(Planck15.age(4).value, Planck15.age(1).value)
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='in')
ax1.tick_params(left=True, right=True)
plt.gca().invert_xaxis()
ax1.set_ylabel('$\dot{\mathrm{M}}_{\mathrm{in}}(\mathrm{R}_{\mathrm{vir}})$ [M$_\odot \mathrm{yr}^{-1}$]')
ax1.set_yscale('log')
ax1.set_yticks([1e-3, 1e-2, 1e-1])
ax1.set_ylim(4e-4, 3e-1)
ax1.legend(frameon = False, fontsize=14, loc='upper left')
cu.add_redshifts_ticks(fig, int(1), int(4), fontsize=22)
plt.savefig('mdot_in_vs_cosmic_time_binned.png')
plt.close()

#Plot mdot_out vs cosmic time with binned data
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(binned_cosmic_time1, binned_mdot_out1, 'b-', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(binned_cosmic_time2, binned_mdot_out2, linestyle='-', color='lime', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.set_xlabel('t [Gyr]')
ax1.set_xticks([2, 3, 4, 5])
ax1.set_xlim(Planck15.age(4).value, Planck15.age(1).value)
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='in')
ax1.tick_params(left=True, right=True)
plt.gca().invert_xaxis()
ax1.set_ylabel('$\dot{\mathrm{M}}_{\mathrm{out}}(\mathrm{R}_{\mathrm{vir}})$ [M$_\odot \mathrm{yr}^{-1}$]')
ax1.set_yscale('log')
ax1.set_yticks([1e-3, 1e-2, 1e-1])
ax1.set_ylim(4e-4, 3e-1)
ax1.legend(frameon = False, fontsize=14, loc='upper left')
cu.add_redshifts_ticks(fig, int(1), int(4), fontsize=22)
plt.savefig('mdot_out_vs_cosmic_time_binned.png')
plt.close()

#Plot beta vs cosmic time with binned data
fig, ax1 = plt.subplots(figsize=(8,6))
plt.tight_layout(pad=3)
ax1.plot(binned_cosmic_time1, binned_beta1, linestyle='-', color='lime', linewidth=3, label='NoAGNHighSN') #NoBHFableHighSNEff
ax1.plot(binned_cosmic_time2, binned_beta2, 'b-', linewidth=3, label='NoAGNHighSNRes') #NoBHFableHighSNEffHighRes
ax1.set_xlabel('t [Gyr]')
ax1.set_xticks([2, 3, 4, 5])
ax1.set_xlim(Planck15.age(4).value, Planck15.age(1).value)
ax1.minorticks_on()
ax1.tick_params(which='minor', length=3, width=1)
ax1.tick_params(which='both', direction='in')
ax1.tick_params(left=True, right=True)
plt.gca().invert_xaxis()
ax1.set_ylabel(r'$\mathrm{\beta}_{\mathrm{out}}(\mathrm{R}_{\mathrm{vir}})$')
ax1.set_yscale('log')
ax1.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
ax1.set_ylim(1e-2, 2e3)
ax1.legend(frameon = False, fontsize=14, loc='upper left')
cu.add_redshifts_ticks(fig, int(1), int(4), fontsize=22)
plt.savefig('beta_vs_cosmic_time_binned.png')
plt.close()