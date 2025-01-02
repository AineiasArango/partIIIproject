def radial_velocities(snap_dir, snap_number, rads, delta_rs):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    from scipy import spatial
    import cosmo_utils as cu
    
    #constants
    h = 0.679 #dimensionless Hubble constant
    
    #get fof file and snap file
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)

    #get host group
    host_groups = cu.get_host_groups(snap_dir, snap_number, 0)
    gal_inds = host_groups == 0

    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)#virial radius (kpc)    
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)
    gal_pos = pos0[gal_inds] #halo gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (Msun)
    gal_mass = mass0[gal_inds]
    v_gas0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True) #velocity of every gas particle in the snapshot (km/s)    
    gal_v_gas = v_gas0[gal_inds]

    #Make the gas tree
    gal_gas_tree = spatial.cKDTree(gal_pos) #only build tree with cells from main halo
    gas_tree = spatial.cKDTree(pos0)

    #Find the mass weighted mean velocity of the gas particles within 2*r_vir of the subhalo
    central_inds = gal_gas_tree.query_ball_point(subhalopos, 2.0*r_vir) #indices of gas particles within 2*r_vir of the subhalo
    central_vels = gal_v_gas[central_inds]
    central_masses = gal_mass[central_inds]
    central_vel = np.average(central_vels, axis=0, weights=central_masses) #mass weighted mean velocity

    v_radials = []
    for rad, delta_r in zip(rads, delta_rs):
        #find appropriate indices
        indices_within_outer_limit = gas_tree.query_ball_point(subhalopos, rad + delta_r/2)
        indices_within_inner_limit = gas_tree.query_ball_point(subhalopos, rad - delta_r/2)
        shell_indices = np.setdiff1d(indices_within_outer_limit, indices_within_inner_limit)

        if shell_indices.size == 0:
            v_radial = 0
        else:
            #shell quantities   
            v_gas_shell = v_gas0[shell_indices] - central_vel #velocity recentred
            pos_shell = pos0[shell_indices] - subhalopos #particle positions
            mass_shell = mass0[shell_indices]    
            
            #work out radial velocity
            norms = np.linalg.norm(pos_shell, axis=1)[:, np.newaxis]
            v_radial_shell = np.einsum('ij,ij->i', v_gas_shell, pos_shell/norms) #radial component of velocity from dot product

            #Find particles with negative radial velocity, no cut-off
            negative_shell_indices = np.where(v_radial_shell < 0)[0]

            if negative_shell_indices.size == 0:
                v_radial = 0
            else:
                #calculate the mass weighted mean radial velocity of the shell
                v_radial = -1*np.average(v_radial_shell[negative_shell_indices], axis=0, weights=mass_shell[negative_shell_indices]) #(km/s)
        v_radials.append(v_radial)
    return np.array(v_radials)

def radial_velocities_T(snap_dir, snap_number, rads, delta_rs, temp_thresh=1e5):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    from scipy import spatial
    from scipy import constants
    import cosmo_utils as cu
    
    #constants
    h = 0.679 #dimensionless Hubble constant
    k_B = constants.k*1e7 #Boltzmann constant (erg/K)
    m_proton = constants.m_p*1e3 #proton mass (g)
    X_H = 0.76 #hydrogen mass fraction
    gamma = 5/3


    #get fof file and snap file
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)

    #get host group
    host_groups = cu.get_host_groups(snap_dir, snap_number, 0)
    gal_inds = host_groups == 0

    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)#virial radius (kpc)    
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)
    gal_pos = pos0[gal_inds] #halo gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (Msun)
    gal_mass = mass0[gal_inds]
    v_gas0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True) #velocity of every gas particle in the snapshot (km/s)    
    gal_v_gas = v_gas0[gal_inds]
    x_e = rsf.get_snap_data(snap_name,0,"ElectronAbundance") #electron abundance, n_e/n_H
    int_energy = rsf.get_snap_data(snap_name,0,"InternalEnergy")*(10**10) #internal energy of every gas particle in the snapshot (cm/s)^2

    #get temperature field of gas particles
    mu = 4*m_proton/(1+3*X_H+4*X_H*x_e)
    T_gas = (gamma-1)*int_energy*mu/k_B #temperature of gas cells (K)

    #Make the gas tree
    gal_gas_tree = spatial.cKDTree(gal_pos) #only build tree with cells from main halo
    gas_tree = spatial.cKDTree(pos0)

    #Find the mass weighted mean velocity of the gas particles within 2*r_vir of the subhalo
    central_inds = gal_gas_tree.query_ball_point(subhalopos, 2.0*r_vir) #indices of gas particles within 2*r_vir of the subhalo
    central_vels = gal_v_gas[central_inds]
    central_masses = gal_mass[central_inds]
    central_vel = np.average(central_vels, axis=0, weights=central_masses) #mass weighted mean velocity

    v_radials_hot = []
    v_radials_cold = []
    for rad, delta_r in zip(rads, delta_rs):
        #find appropriate indices
        indices_within_outer_limit = gas_tree.query_ball_point(subhalopos, rad + delta_r/2)
        indices_within_inner_limit = gas_tree.query_ball_point(subhalopos, rad - delta_r/2)
        shell_indices = np.setdiff1d(indices_within_outer_limit, indices_within_inner_limit)

        if shell_indices.size == 0:
            v_radial_hot = 0
            v_radial_cold = 0
        else:
            #shell quantities   
            v_gas_shell = v_gas0[shell_indices] - central_vel #velocity recentred
            pos_shell = pos0[shell_indices] - subhalopos #particle positions
            mass_shell = mass0[shell_indices]
            T_shell = T_gas[shell_indices]
            
            #work out radial velocity
            norms = np.linalg.norm(pos_shell, axis=1)[:, np.newaxis]
            v_radial_shell = np.einsum('ij,ij->i', v_gas_shell, pos_shell/norms) #radial component of velocity from dot product

            #Find particles with negative radial velocity, no cut-off, and split into hot and cold
            negative_shell_indices_hot = np.where((v_radial_shell < 0) & (T_shell > temp_thresh))[0]
            negative_shell_indices_cold = np.where((v_radial_shell < 0) & (T_shell <= temp_thresh))[0]

            if negative_shell_indices_hot.size == 0:
                v_radial_hot = 0
            else:
                #calculate the mass weighted mean radial velocity of the shell
                v_radial_hot = -1*np.average(v_radial_shell[negative_shell_indices_hot], axis=0, weights=mass_shell[negative_shell_indices_hot])
            if negative_shell_indices_cold.size == 0:
                v_radial_cold = 0
            else:
                v_radial_cold = -1*np.average(v_radial_shell[negative_shell_indices_cold], axis=0, weights=mass_shell[negative_shell_indices_cold])

        v_radials_hot.append(v_radial_hot)
        v_radials_cold.append(v_radial_cold)
    return np.array(v_radials_hot), np.array(v_radials_cold)

def radial_velocity_function(snap_dir, snap_number, rads, delta_rs, split_temp=False, temp_thresh=10**4.25):
    if split_temp:
        return radial_velocities_T(snap_dir, snap_number, rads, delta_rs, temp_thresh=temp_thresh)
    else:
        return radial_velocities(snap_dir, snap_number, rads, delta_rs)