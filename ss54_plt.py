import numpy as np
import matplotlib.pyplot as plt
import cosmo_utils as cu
import os
import read_fof_files as rff
from scipy import constants
from plotting_function import radial_plotting_function

snap_number = 54
os.chdir("/home/aasnha2/Project/Plots/ss" + str(snap_number))

#get the virial radius of the particular snapshot
snap_dir1 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
snap_dir2 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"
fof_file = rff.get_fof_filename(snap_dir2, snap_number)
a = rff.get_attribute(fof_file, "Time") #scale factor
h = 0.679 #dimensionless Hubble constant
r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)*1000 #virial radius (pc)    
Msun = 1.988416e30 #kg
Redshift = rff.get_attribute(fof_file, "Redshift")
print(Redshift)
r_soft1 = cu.cosmo_to_phys(0.387, a, h, length=True)*1000 #softening length (pc)
r_soft2 = cu.cosmo_to_phys(0.194, a, h, length=True)*1000 #softening length (pc)

#load the data
#mass inflow rate
data = np.load('ss'+str(snap_number)+'_mass_inflow_rate_data.npz')
rads = data['rads']*1000 #convert to pc
mdot_ins1 = data['mdot_ins1']
mdot_ins2 = data['mdot_ins2']
#mass inflow rate with temp split
data = np.load('ss'+str(snap_number)+'_mass_inflow_rate_T_data.npz')
mdot_ins1_hot = data['mdot_ins1_hot']
mdot_ins1_cold = data['mdot_ins1_cold']
mdot_ins2_hot = data['mdot_ins2_hot']
mdot_ins2_cold = data['mdot_ins2_cold']
#mass outflow rate
data = np.load('ss'+str(snap_number)+'_mass_outflow_rate_data.npz')
mdot_outs1 = data['mdot_outs1']
mdot_outs2 = data['mdot_outs2']
#mass outflow rate with temp split
data = np.load('ss'+str(snap_number)+'_mass_outflow_rate_T_data.npz')
mdot_outs1_hot = data['mdot_outs1_hot']
mdot_outs1_cold = data['mdot_outs1_cold']
mdot_outs2_hot = data['mdot_outs2_hot']
mdot_outs2_cold = data['mdot_outs2_cold']
#internal mass
data = np.load('ss'+str(snap_number)+'_mass_data.npz')
masses1 = data['masses1']
masses2 = data['masses2']
#internal mass with temp split
data = np.load('ss'+str(snap_number)+'_mass_data_temp_split.npz')
masses_hot1 = data['masses_hot1']
masses_cold1 = data['masses_cold1']
masses_hot2 = data['masses_hot2']
masses_cold2 = data['masses_cold2']
#total internal mass + work out keplerian velocity
data = np.load('ss'+str(snap_number)+'_total_mass_data.npz')
total_masses2 = data['total_masses2']
unit_conversion = Msun/3.0857e16
v_ks2 = np.sqrt(constants.G*total_masses2*unit_conversion/rads)/1000
#mass density
data = np.load('ss'+str(snap_number)+'_mass_density_data.npz')
mass_densities1 = data['mass_densities1']
mass_densities2 = data['mass_densities2']
#mass density with temp split
data = np.load('ss'+str(snap_number)+'_mass_density_data_temp_split.npz')
mass_densities_hot1 = data['mass_densities_hot1']
mass_densities_cold1 = data['mass_densities_cold1']
mass_densities_hot2 = data['mass_densities_hot2']
mass_densities_cold2 = data['mass_densities_cold2']
#radial velocity
data = np.load('ss'+str(snap_number)+'_radial_velocity_data.npz')
v_radials1 = data['v_radials1']
v_radials2 = data['v_radials2']
#radial velocity with temp split
data = np.load('ss'+str(snap_number)+'_radial_velocity_T_data.npz')
v_radials_hot1 = data['v_radials_hot1']
v_radials_hot2 = data['v_radials_hot2']
v_radials_cold1 = data['v_radials_cold1']
v_radials_cold2 = data['v_radials_cold2']
#angular momentum
data = np.load('ss'+str(snap_number)+'_ang_mom_data.npz')
ang_mom_mags_hot1 = data['ang_mom_mags_hot1']
ang_mom_mags_hot2 = data['ang_mom_mags_hot2']
ang_mom_mags_cold1 = data['ang_mom_mags_cold1']
ang_mom_mags_cold2 = data['ang_mom_mags_cold2']
#total angular momentum
data = np.load('ss'+str(snap_number)+'_tot_ang_mom_data.npz')
tot_ang_mom_mags_hot1 = data['tot_ang_mom_mags_hot1']
tot_ang_mom_mags_hot2 = data['tot_ang_mom_mags_hot2']
tot_ang_mom_mags_cold1 = data['tot_ang_mom_mags_cold1']
tot_ang_mom_mags_cold2 = data['tot_ang_mom_mags_cold2']

r_vir_label = False

save_dir = '/home/aasnha2/Project/Plots/ss' + str(snap_number)
#radial velocity
radial_plotting_function(ydata=[v_radials1, v_radials2], xdata=rads, data_labels=['NoAGNHighSN', 'NoAGNHighSNRes'], xlabel='r [pc]', 
                         ylabel='v$_{\mathrm{r}}$ [km/s]', r_vir=r_vir, save_name='ss'+str(snap_number)+'_v_radial_vs_r.png', save_dir=save_dir, 
                         v_k=v_ks2, r_soft1=r_soft1, r_soft2=r_soft2, r_vir_label=r_vir_label)
#radial velocity with temp split
radial_plotting_function(ydata=[v_radials_hot1, v_radials_hot2, v_radials_cold1, v_radials_cold2], xdata=rads, 
                         data_labels=['NoAGNHighSN (hot)', 'NoAGNHighSNRes (hot)', 'NoAGNHighSN (cold)', 'NoAGNHighSNRes (cold)'], xlabel='r [pc]', 
                         ylabel='v$_{\mathrm{r}}$ [km/s]', temp_split=True, r_vir=r_vir, save_name='ss'+str(snap_number)+'_v_radial_T_vs_r.png', 
                         save_dir=save_dir, v_k=v_ks2, r_soft1=r_soft1, r_soft2=r_soft2, r_vir_label=r_vir_label)
#mass inflow rate
radial_plotting_function(ydata=[mdot_ins1, mdot_ins2], xdata=rads, data_labels=['NoAGNHighSN', 'NoAGNHighSNRes'], xlabel='r [pc]', 
                         ylabel='$\dot{\mathrm{M}}_{\mathrm{in}}$ [M$_\odot \mathrm{yr}^{-1}$]', r_vir=r_vir, save_name='ss'+str(snap_number)+'_mdot_in_vs_r.png', 
                         save_dir=save_dir, r_soft1=r_soft1, r_soft2=r_soft2, r_vir_label=r_vir_label)
#mass inflow rate with temp split
radial_plotting_function(ydata=[mdot_ins1_hot, mdot_ins2_hot, mdot_ins1_cold, mdot_ins2_cold], xdata=rads, 
                         data_labels=['NoAGNHighSN (hot)', 'NoAGNHighSNRes (hot)', 'NoAGNHighSN (cold)', 'NoAGNHighSNRes (cold)'], xlabel='r [pc]', 
                         ylabel='$\dot{\mathrm{M}}_{\mathrm{in}}$ [M$_\odot \mathrm{yr}^{-1}$]', temp_split=True, r_vir=r_vir, 
                         save_name='ss'+str(snap_number)+'_mdot_in_T_vs_r.png', save_dir=save_dir, r_soft1=r_soft1, r_soft2=r_soft2, r_vir_label=r_vir_label)
#mass density
radial_plotting_function(ydata=[mass_densities1, mass_densities2], xdata=rads, data_labels=['NoAGNHighSN', 'NoAGNHighSNRes'], xlabel='r [pc]', 
                         ylabel='Shell mass [M$_\odot$/pc]', r_vir=r_vir, save_name='ss'+str(snap_number)+'_mass_density_vs_r.png', 
                         save_dir=save_dir, r_soft1=r_soft1, r_soft2=r_soft2, r_vir_label=r_vir_label)
#mass density with temp split
radial_plotting_function(ydata=[mass_densities_hot1, mass_densities_hot2, mass_densities_cold1, mass_densities_cold2], xdata=rads, 
                         data_labels=['NoAGNHighSN (hot)', 'NoAGNHighSNRes (hot)', 'NoAGNHighSN (cold)', 'NoAGNHighSNRes (cold)'], xlabel='r [pc]', 
                         ylabel='Shell mass [M$_\odot$/pc]', temp_split=True, r_vir=r_vir, 
                         save_name='ss'+str(snap_number)+'_mass_density_T_vs_r.png', save_dir=save_dir, r_soft1=r_soft1, r_soft2=r_soft2, r_vir_label=r_vir_label)
#internal mass
radial_plotting_function(ydata=[masses1, masses2], xdata=rads, data_labels=['NoAGNHighSN', 'NoAGNHighSNRes'], xlabel='r [pc]', 
                         ylabel='M$_{\mathrm{gas}}(\mathrm{r})$ [M$_\odot$]', r_vir=r_vir, save_name='ss'+str(snap_number)+'_internal_mass_vs_r.png', 
                         save_dir=save_dir, r_soft1=r_soft1, r_soft2=r_soft2, r_vir_label=r_vir_label)
#internal mass with temp split
radial_plotting_function(ydata=[masses_hot1, masses_hot2, masses_cold1, masses_cold2], xdata=rads, 
                         data_labels=['NoAGNHighSN (hot)', 'NoAGNHighSNRes (hot)', 'NoAGNHighSN (cold)', 'NoAGNHighSNRes (cold)'], xlabel='r [pc]', 
                         ylabel='M$_{\mathrm{gas}}(\mathrm{r})$ [M$_\odot$]', temp_split=True, r_vir=r_vir, 
                         save_name='ss'+str(snap_number)+'_internal_mass_T_vs_r.png', save_dir=save_dir, r_soft1=r_soft1, r_soft2=r_soft2, r_vir_label=r_vir_label)
#angular momentum
radial_plotting_function(ydata=[ang_mom_mags_hot1, ang_mom_mags_hot2, ang_mom_mags_cold1, ang_mom_mags_cold2], xdata=rads, 
                         data_labels=['NoAGNHighSN (hot)', 'NoAGNHighSNRes (hot)', 'NoAGNHighSN (cold)', 'NoAGNHighSNRes (cold)'], xlabel='r [pc]', 
                         ylabel='Angular momentum [M$_\odot$ pc km/s]', temp_split=True, r_vir=r_vir, 
                         save_name='ss'+str(snap_number)+'_ang_mom_T_vs_r.png', save_dir=save_dir, r_soft1=r_soft1, r_soft2=r_soft2, r_vir_label=r_vir_label)
#total angular momentum
radial_plotting_function(ydata=[tot_ang_mom_mags_hot1, tot_ang_mom_mags_hot2, tot_ang_mom_mags_cold1, tot_ang_mom_mags_cold2], xdata=rads, 
                         data_labels=['NoAGNHighSN (hot)', 'NoAGNHighSNRes (hot)', 'NoAGNHighSN (cold)', 'NoAGNHighSNRes (cold)'], xlabel='r [pc]', 
                         ylabel='Total angular momentum [M$_\odot$ pc km/s]', temp_split=True, r_vir=r_vir, 
                         save_name='ss'+str(snap_number)+'_tot_ang_mom_T_vs_r.png', save_dir=save_dir, r_soft1=r_soft1, r_soft2=r_soft2, r_vir_label=r_vir_label)