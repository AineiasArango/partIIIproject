# for the main galaxy/subhalo (i.e. subhalo 0)
import matplotlib.pyplot as plt
import numpy as np
import read_fof_files as rff
import read_snap_files as rsf
import sys
sys.path.insert(0, "/data/ERCblackholes4/sk939/for_aineias")

#function calculates the mass flow rate in Msun/yr at the virial radius for a single snapshot
def mass_flow_rate(snap_dir, snap_number, delta_r=10):
    #constants
    h = 0.679 #dimensionless Hubble constant
    seconds_per_year = 31557600 #number of seconds in a year
    
    #get fof file and snap file
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)

    #get attributes
    redshift = rff.get_attribute(fof_file, "Redshift") #redshift
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = rff.get_subhalo_data(fof_file, 'SubhaloPos')[0]*a #subhalo position
    r_vir = rff.get_group_data(fof_file, "Group_R_Crit200")[0]*a #virial radius    
    pos0 = rsf.get_snap_data(snap_name,0,"Coordinates")*a #gas particle positions
    mass0 = rsf.get_snap_data(snap_name,0,"Masses") #gas particle masses
    sfr = rsf.get_snap_data(snap_name,0,"StarFormationRate") #gas cell sfr
    v_gas = rsf.get_snap_data(snap_name,0,"Velocities")*np.sqrt(a) #velocity of every gas particle in the snapshot

    #recentre coorinates
    pos0_c = pos0 - subhalopos #recentre the position of every gas particle on the subhalo
    r = np.sqrt(np.sum((pos0_c)**2, axis=1)) #radial distance of each particle
    
    #prepare to recentre velocity
    indices_within_two_r_vir = np.where(r < 2*r_vir)[0] #indices of gas particles within twice the virial radius
    mass_weighted_mean_velocity = np.sum(mass0[indices_within_two_r_vir][:, np.newaxis]*v_gas[indices_within_two_r_vir])/np.sum(mass0[indices_within_two_r_vir]) #mass-weighted mean velocity of gas particles within delta_r of the virial radius

    #shell quantities
    shell_indices = np.where((r >= r_vir - delta_r/2) & (r < r_vir + delta_r/2))[0] #indices of gas particles within delta_r of the virial radius
    indices_within_one_r_vir = np.where(r < r_vir)[0] #indices of gas particles within the virial radius    
    v_gas_shell = v_gas[shell_indices] - mass_weighted_mean_velocity #velocity recentred
    pos_shell = pos0_c[shell_indices] #particle positions
    mass_shell = mass0[shell_indices]    
    sfr_within_one_r_vir = np.sum(sfr[indices_within_one_r_vir]) #sfr of gas particles within the virial radius
    
    #work out radial velocity
    pos_shell_normalized = pos_shell / r[shell_indices][:, np.newaxis]
    v_radial_shell = np.sum(v_gas_shell * pos_shell_normalized, axis=1) #radial component of velocity from dot product

    #Find particles with positive and negative radial velocity
    positive_shell_indices = np.where(v_radial_shell > 0)[0]
    negative_shell_indices = np.where(v_radial_shell < 0)[0]

    #Calculate mass flow rates in Msun/yr
    unit_conversion = 10**10 *seconds_per_year*3.24*10**(-17)/h #converting from 10^10 Msun/h * km/s /kpc to Msun/yr
    mdot_out = np.sum(mass_shell[positive_shell_indices]*v_radial_shell[positive_shell_indices]/delta_r)*unit_conversion #out
    mdot_in = np.sum(mass_shell[negative_shell_indices]*v_radial_shell[negative_shell_indices]/delta_r)*unit_conversion #in - note this value is negative
    mdot_tot = mdot_out + mdot_in #tot
    beta = mdot_out/sfr_within_one_r_vir #mass loading factor   
    return [redshift, mdot_tot, mdot_in, mdot_out, beta]

