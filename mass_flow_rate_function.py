# for the main galaxy/subhalo (i.e. subhalo 0)
import matplotlib.pyplot as plt
import numpy as np
import read_fof_files as rff
import read_snap_files as rsf
import sys
from scipy import spatial
import cosmo_utils as cu
sys.path.insert(0, "/data/ERCblackholes4/sk939/for_aineias")

#function calculates the mass flow rate in Msun/yr at the virial radius for a single snapshot
def mass_flow_rate(snap_dir, snap_number, delta_r=3):
    #constants
    h = 0.679 #dimensionless Hubble constant
    
    #get fof file and snap file
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)

    #get host group
    host_groups = cu.get_host_groups(snap_dir, snap_number, 0)
    gal_inds = host_groups == 0

    #get attributes and convert units
    redshift = rff.get_attribute(fof_file, "Redshift") #redshift
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)#virial radius (kpc)    
    pos0 = rsf.get_snap_data(snap_name,0,"Coordinates") #gas particle positions (kpc)
    gal_pos = cu.cosmo_to_phys(pos0[gal_inds], a, h, length=True) #halo gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (Msun)
    sfr = rsf.get_snap_data(snap_name,0,"StarFormationRate") #gas cell sfr (Msun/yr)
    v_gas = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True) #velocity of every gas particle in the snapshot (km/s)    

    #Make the gas tree
    gas_tree = spatial.cKDTree(gal_pos) #only build tree with cells from main halo

    #Find the mass weighted mean velocity of the gas particles within 2*r_vir of the subhalo
    central_inds = gas_tree.query_ball_point(subhalopos, 2.0*r_vir) #indices of gas particles within 2*r_vir of the subhalo
    central_vels = v_gas[central_inds]
    central_masses = mass0[central_inds]
    central_vel = np.average(central_vels, axis=0, weights=central_masses) #mass weighted mean velocity

    #find appropriate indices
    indices_within_r_vir = gas_tree.query_ball_point(subhalopos, r_vir) #indices of gas particles within r_vir of the subhalo
    indices_within_outer_limit = gas_tree.query_ball_point(subhalopos, r_vir + delta_r/2)
    indices_within_inner_limit = gas_tree.query_ball_point(subhalopos, r_vir - delta_r/2)
    shell_indices = np.setdiff1d(indices_within_outer_limit, indices_within_inner_limit)

    #shell quantities   
    v_gas_shell = v_gas[shell_indices] - central_vel #velocity recentred
    pos_shell = gal_pos[shell_indices] - subhalopos #particle positions
    mass_shell = mass0[shell_indices]    
    sfr_within_r_vir = np.sum(sfr[indices_within_r_vir]) #sfr of gas particles within the virial radius
    
    #work out radial velocity
    norms = np.linalg.norm(pos_shell, axis=1)[:, np.newaxis]
    v_radial_shell = np.einsum('ij,ij->i', v_gas_shell, pos_shell/norms) #radial component of velocity from dot product

    #Find particles with positive and negative radial velocity, including a cutoff at +- 1 km/s
    positive_shell_indices = np.where(v_radial_shell > 1)[0]
    negative_shell_indices = np.where(v_radial_shell < -1)[0]

    #Calculate mass flow rates in Msun/yr
    unit_conversion = 31557600*3.24078e-17 #yr/s*km/kpc
    mdot_out = np.sum(mass_shell[positive_shell_indices]*v_radial_shell[positive_shell_indices]/delta_r)*unit_conversion #out #Msun/yr
    mdot_in = np.sum(mass_shell[negative_shell_indices]*v_radial_shell[negative_shell_indices]/delta_r)*unit_conversion #in - note this value is negative
    mdot_tot = mdot_out + mdot_in #tot
    beta = mdot_out/sfr_within_r_vir #mass loading factor   
    return [redshift, mdot_tot, mdot_in, mdot_out, beta]

