# for the main galaxy/subhalo (i.e. subhalo 0)
import matplotlib.pyplot as plt
import numpy as np
import read_fof_files as rff
import read_snap_files as rsf
import sys
sys.path.insert(0, "/data/ERCblackholes4/sk939/for_aineias")

#function calculates the mass flow rate in Msun/yr at the virial radius for a single snapshot
def mass_flow_rate(snap_dir, snap_number, delta_r=10):
    h = 0.679 #dimensionless Hubble constant
    seconds_per_year = 31557600 #number of seconds in a year
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    redshift = rff.get_attribute(fof_file, "Redshift") #redshift
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = rff.get_subhalo_data(fof_file, 'SubhaloPos')[0]*a #subhalo position
    r_vir = rff.get_group_data(fof_file, "Group_R_Crit200")[0]*a #get the virial radius of the group. Is this the virial radius of the subhalo we are analysing?
    
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)
    pos0 = rsf.get_snap_data(snap_name,0,"Coordinates")*a #use 0 because gas is particle type 0. Position of every gas particle in the snapshot
    mass0 = rsf.get_snap_data(snap_name,0,"Masses") #mass of every gas particle in the snapshot
    sfr = rsf.get_snap_data(snap_name,0,"StarFormationRate") #star formation rate of every gas particle in the snapshot
    pos0_c = pos0 - subhalopos #recentre the position of every gas particle on the subhalo
    r = np.sqrt(np.sum((pos0_c)**2, axis=1)) #distance of each particle from centre of subhalo
    #In the paper, they define the peculiar velocity as the mass-weighted mean velocity of all gas cells within twice the virial radius. But, on the illustris webpage, it says the peculiar velocity is obtained by multiplying the velocity value by sqrt(a)
    #Here, we will use what was done in the paper.
    #For now, we will just track the gas.
    v_gas = rsf.get_snap_data(snap_name,0,"Velocities") #velocity of every gas particle in the snapshot
    shell_indices = np.where((r >= r_vir - delta_r/2) & (r < r_vir + delta_r/2))[0] #indices of gas particles within delta_r of the virial radius
    indices_within_two_r_vir = np.where(r < 2*r_vir)[0] #indices of gas particles within twice the virial radius
    indices_within_one_r_vir = np.where(r < r_vir)[0] #indices of gas particles within the virial radius
    mass_weighted_mean_velocity = np.sum(mass0[indices_within_two_r_vir][:, np.newaxis]*v_gas[indices_within_two_r_vir])/np.sum(mass0[indices_within_two_r_vir]) #mass-weighted mean velocity of gas particles within delta_r of the virial radius
    sfr_within_one_r_vir = np.sum(sfr[indices_within_one_r_vir]) #sfr of gas particles within the virial radius

    v_gas_shell = v_gas[shell_indices] - mass_weighted_mean_velocity
    #If we were using the peculiar velocity from the webpage, we would do the following:
    #v_gas_shell = v_gas[shell_indices] - v_gas[shell_indices]*np.sqrt(a) - mass_weighted_mean_velocity   

    # Get position vectors for particles in the shell
    pos_shell = pos0_c[shell_indices]
    # Normalize the position vectors
    pos_shell_normalized = pos_shell / r[shell_indices][:, np.newaxis]
    # Take dot product of each particle's velocity with its normalized position vector, revealing a list of radial speeds
    v_radial_shell = np.sum(v_gas_shell * pos_shell_normalized, axis=1)
    #We can now work out the mass accretion rate by integrating the mass weighted radial velocity over the shell
    masses_shell = mass0[shell_indices]
    #Find particles with positive radial velocity
    positive_shell_indices = np.where(v_radial_shell > 0)[0]
    negative_shell_indices = np.where(v_radial_shell < 0)[0]
    #Calculate outward going mass flow rate
    mdot_out = np.sum(masses_shell[positive_shell_indices]*v_radial_shell[positive_shell_indices]/delta_r) #units of 10^10 Msun/h * km/s /kpc
    #multiply by 10^10 *31557600 km/kpc /h to get Msun/yr
    mdot_out *= 10**10 *seconds_per_year*3.24*10**(-17)/h
    #Calculate inward going mass flow rate
    mdot_in = np.sum(masses_shell[negative_shell_indices]*v_radial_shell[negative_shell_indices]/delta_r)
    mdot_in *= 10**10 *seconds_per_year*3.24*10**(-17)/h
    #Calculate net mass flow rate
    mdot_tot = mdot_out + mdot_in #units of Msun/yr
    beta = mdot_out/sfr_within_one_r_vir #mass loading factor   
    return [redshift, mdot_tot, mdot_in, mdot_out, beta]
