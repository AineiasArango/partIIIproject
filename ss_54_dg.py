import numpy as np
import read_fof_files as rff
import cosmo_utils as cu
import os
import mass_functions as mf
import mass_flow_rate_functions as mfrf
import radial_velocity_functions as rvf

snap_dir1="/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
snap_dir2="/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"

for i in [54, 86]:
    snap_number = i
    os.chdir("/home/aasnha2/Project/Plots/ss"+str(snap_number))

    #Calculate i_max using the virial radius
    #We want r to range from r_0 to r_vir
    fof_file = rff.get_fof_filename(snap_dir1, snap_number)
    a = rff.get_attribute(fof_file, "Time") #scale factor
    h = 0.679 #dimensionless Hubble constant
    r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)#virial radius (kpc)    
    r_0 = 1e-4 #kpc
    temp_thresh = 10**4.25 #K

    #Produce rads and delta_rs
    #Done so that the bins touch and the bins are 10% the width of the radius
    ratio = 0.1
    fraction = (1+ratio/2)/(1-ratio/2)
    i_max = int(np.log(3*r_vir/r_0)/np.log(fraction)) + 1
    rads = r_0*np.logspace(0, i_max, num = i_max + 1, base = fraction)
    delta_rs = ratio*rads

    #Mass profile
    #internal mass no temp split
    masses1 = mf.mass_function(snap_dir1, snap_number, rads)
    masses2 = mf.mass_function(snap_dir2, snap_number, rads)
    np.savez('ss'+str(snap_number)+'_mass_data.npz', masses1=masses1, masses2=masses2, rads=rads)
    #internal mass temp split
    masses_hot1, masses_cold1 = mf.mass_function(snap_dir1, snap_number, rads, split_temp=True, temp_thresh = temp_thresh)
    masses_hot2, masses_cold2 = mf.mass_function(snap_dir2, snap_number, rads, split_temp=True, temp_thresh = temp_thresh)
    np.savez('ss'+str(snap_number)+'_mass_data_temp_split.npz', masses_hot1=masses_hot1, masses_cold1=masses_cold1, masses_hot2=masses_hot2, masses_cold2=masses_cold2, rads=rads)
    #mass density no temp split
    mass_densities1 = mf.mass_density_function(snap_dir1, snap_number, rads, delta_rs)
    mass_densities2 = mf.mass_density_function(snap_dir2, snap_number, rads, delta_rs)
    np.savez('ss'+str(snap_number)+'_mass_density_data.npz', mass_densities1=mass_densities1, mass_densities2=mass_densities2, rads=rads)
    #mass density temp split
    mass_densities_hot1, mass_densities_cold1 = mf.mass_density_function(snap_dir1, snap_number, rads, delta_rs, split_temp=True, temp_thresh = temp_thresh)
    mass_densities_hot2, mass_densities_cold2 = mf.mass_density_function(snap_dir2, snap_number, rads, delta_rs, split_temp=True, temp_thresh = temp_thresh)
    np.savez('ss'+str(snap_number)+'_mass_density_data_temp_split.npz', mass_densities_hot1=mass_densities_hot1, mass_densities_cold1=mass_densities_cold1, mass_densities_hot2=mass_densities_hot2, mass_densities_cold2=mass_densities_cold2, rads=rads)
    #total mass
    total_masses1 = mf.total_mass_function(snap_dir1, snap_number, rads, 1536)
    total_masses2 = mf.total_mass_function(snap_dir2, snap_number, rads, 192)
    np.savez('ss'+str(snap_number)+'_total_mass_data.npz', total_masses1=total_masses1, total_masses2=total_masses2, rads=rads)

    #Mass flow rate
    mdot_ins1 = np.array(mfrf.mass_flow_rate_function(snap_dir1, snap_number, rads, delta_rs, split_temp=False))
    mdot_ins2 = np.array(mfrf.mass_flow_rate_function(snap_dir2, snap_number, rads, delta_rs, split_temp=False))
    np.savez('ss'+str(snap_number)+'_mass_inflow_rate_data.npz', mdot_ins1=mdot_ins1, mdot_ins2=mdot_ins2, rads=rads)
    mdot_outs1 = np.array(mfrf.mass_flow_rate_function(snap_dir1, snap_number, rads, delta_rs, split_temp=False, inflow=False))
    mdot_outs2 = np.array(mfrf.mass_flow_rate_function(snap_dir2, snap_number, rads, delta_rs, split_temp=False, inflow=False))
    np.savez('ss'+str(snap_number)+'_mass_outflow_rate_data.npz', mdot_outs1=mdot_outs1, mdot_outs2=mdot_outs2, rads=rads)
    #temp split
    mdot_ins1_hot, mdot_ins1_cold = np.array(mfrf.mass_flow_rate_function(snap_dir1, snap_number, rads, delta_rs, split_temp=True, temp_thresh = temp_thresh))
    mdot_ins2_hot, mdot_ins2_cold = np.array(mfrf.mass_flow_rate_function(snap_dir2, snap_number, rads, delta_rs, split_temp=True, temp_thresh = temp_thresh))
    np.savez('ss'+str(snap_number)+'_mass_inflow_rate_T_data.npz', mdot_ins1_hot=mdot_ins1_hot, mdot_ins1_cold=mdot_ins1_cold, mdot_ins2_hot=mdot_ins2_hot, mdot_ins2_cold=mdot_ins2_cold, rads=rads)
    mdot_outs1_hot, mdot_outs1_cold = np.array(mfrf.mass_flow_rate_function(snap_dir1, snap_number, rads, delta_rs, split_temp=True, inflow=False, temp_thresh = temp_thresh))
    mdot_outs2_hot, mdot_outs2_cold = np.array(mfrf.mass_flow_rate_function(snap_dir2, snap_number, rads, delta_rs, split_temp=True, inflow=False, temp_thresh = temp_thresh))
    np.savez('ss'+str(snap_number)+'_mass_outflow_rate_T_data.npz', mdot_outs1_hot=mdot_outs1_hot, mdot_outs1_cold=mdot_outs1_cold, mdot_outs2_hot=mdot_outs2_hot, mdot_outs2_cold=mdot_outs2_cold, rads=rads)

    #Radial velocity
    v_radials1 = rvf.radial_velocity_function(snap_dir1, snap_number, rads, delta_rs)
    v_radials2 = rvf.radial_velocity_function(snap_dir2, snap_number, rads, delta_rs)
    np.savez('ss'+str(snap_number)+'_radial_velocity_data.npz', v_radials1=v_radials1, v_radials2=v_radials2, rads=rads)

    v_radials_hot1, v_radials_cold1 = rvf.radial_velocity_function(snap_dir1, snap_number, rads, delta_rs, split_temp = True, temp_thresh = temp_thresh)
    v_radials_hot2, v_radials_cold2 = rvf.radial_velocity_function(snap_dir2, snap_number, rads, delta_rs, split_temp = True, temp_thresh = temp_thresh)
    np.savez('ss'+str(snap_number)+'_radial_velocity_T_data.npz', v_radials_hot1=v_radials_hot1, v_radials_hot2=v_radials_hot2, v_radials_cold1=v_radials_cold1, v_radials_cold2=v_radials_cold2, rads=rads)
    
    print("Done with ss"+str(snap_number))
