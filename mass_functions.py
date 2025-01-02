def mass_function(snap_dir, snap_number, rads, split_temp = False, temp_thresh = 10**4.25):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    from scipy import spatial
    import cosmo_utils as cu
    from scipy import constants

    #constants
    h = 0.679 #dimensionless Hubble constant
    k_B = constants.k*1e7 #Boltzmann constant (erg/K)
    m_proton = constants.m_p*1e3 #proton mass (g)
    X_H = 0.76 #hydrogen mass fraction
    gamma = 5/3

    #get fof file and snap file
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)

    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)  
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (Msun
    x_e = rsf.get_snap_data(snap_name,0,"ElectronAbundance") #electron abundance, n_e/n_H
    int_energy = rsf.get_snap_data(snap_name,0,"InternalEnergy")*(10**10) #internal energy of every gas particle in the snapshot (cm/s)^2

    #Make the gas tree
    gas_tree = spatial.cKDTree(pos0)
    #if user wants to split the gas up into hot and cold gas with the threshold being at temp_thresh
    if split_temp:
        #get temperature field of gas particles
        mu = 4*m_proton/(1+3*X_H+4*X_H*x_e)
        T_gas = (gamma-1)*int_energy*mu/k_B #temperature of gas cells (K)
        masses_hot = []
        masses_cold = []
        for rad in rads:
            indices_within_r = gas_tree.query_ball_point(subhalopos, rad)
            if len(indices_within_r) == 0:
                masses_hot.append(0)
                masses_cold.append(0)
            else:
                masses_within_r = mass0[indices_within_r]
                T_within_r = T_gas[indices_within_r]
                hot_inds = np.where(T_within_r > temp_thresh)[0]
                cold_inds = np.where(T_within_r <= temp_thresh)[0]
                if hot_inds.size == 0:
                    masses_hot.append(0)
                else:
                    masses_hot.append(np.sum(masses_within_r[hot_inds]))
                if cold_inds.size == 0:
                    masses_cold.append(0)
                else:
                    masses_cold.append(np.sum(masses_within_r[cold_inds]))
        return np.array(masses_hot), np.array(masses_cold)
    
    else:
        masses = [] 
        for rad in rads:
            indices_within_r = gas_tree.query_ball_point(subhalopos, rad)
            masses_within_r = mass0[indices_within_r]
            mass_within_r = np.sum(masses_within_r)
            masses.append(mass_within_r)

        return np.array(masses)

def dm_mass_function(snap_dir, snap_number, rads, dm_part_mass):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    from scipy import spatial
    import cosmo_utils as cu
    from scipy import constants

    #constants
    h = 0.679 #dimensionless Hubble constant

    #get fof file and snap file
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)

    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)  
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,1,"Coordinates"), a, h, length=True) #dark matter particle positions (kpc)

    #Make the gas tree
    gas_tree = spatial.cKDTree(pos0) #only build tree with cells from main halo

    masses = [] 
    for rad in rads:
        indices_within_r = gas_tree.query_ball_point(subhalopos, rad)
        masses_within_r = dm_part_mass*len(indices_within_r)
        mass_within_r = np.sum(masses_within_r)
        masses.append(mass_within_r)

    return np.array(masses)

def star_mass_function(snap_dir, snap_number, rads):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    from scipy import spatial
    import cosmo_utils as cu
    from scipy import constants

    #constants
    h = 0.679 #dimensionless Hubble constant

    #get fof file and snap file
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)

    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)  
    pos0 = rsf.get_snap_data(snap_name,4,"Coordinates") #gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,4,"Masses"), a, h, mass=True) #gas particle masses (Msun)

    #Make the gas tree
    gas_tree = spatial.cKDTree(pos0) #only build tree with cells from main halo
    
    masses = [] 
    for rad in rads:
        indices_within_r = gas_tree.query_ball_point(subhalopos, rad)
        masses_within_r = mass0[indices_within_r]
        mass_within_r = np.sum(masses_within_r)
        masses.append(mass_within_r)

    return np.array(masses)

def total_mass_function(snap_dir, snap_number, rads, dm_part_mass):
    import numpy as np

    #calculate gas masses
    gas_masses = mass_function(snap_dir, snap_number, rads, split_temp = False)
    dm_masses = dm_mass_function(snap_dir, snap_number, rads, dm_part_mass)
    star_masses = star_mass_function(snap_dir, snap_number, rads)

    #add masses together
    total_masses = gas_masses + dm_masses + star_masses
    return total_masses


def mass_density_function(snap_dir, snap_number, rads, delta_rs, split_temp = False, temp_thresh = 10**4.25):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    from scipy import spatial
    import cosmo_utils as cu
    from scipy import constants

    #constants
    h = 0.679 #dimensionless Hubble constant

    #get fof file and snap file
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)

    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)  
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (Msun)

    #Make the gas tree
    gas_tree = spatial.cKDTree(pos0) #only build tree with cells from main halo

    #if user wants to split the gas up into hot and cold gas with the threshold being at temp_thresh
    if split_temp:
        #constants
        k_B = constants.k*1e7 #Boltzmann constant (erg/K)
        m_proton = constants.m_p*1e3 #proton mass (g)
        X_H = 0.76 #hydrogen mass fraction
        gamma = 5/3

        #get electron abundance and internal energy
        x_e = rsf.get_snap_data(snap_name,0,"ElectronAbundance") #electron abundance, n_e/n_H
        int_energy = rsf.get_snap_data(snap_name,0,"InternalEnergy")*(10**10) #internal energy of every gas particle in the snapshot (cm/s)^2

        #get temperature field of gas particles
        mu = 4*m_proton/(1+3*X_H+4*X_H*x_e)
        T_gas = (gamma-1)*int_energy*mu/k_B #temperature of gas cells (K)
        masses_hot = []
        masses_cold = []                
        for rad, delta_r in zip(rads, delta_rs):
            #find appropriate indices
            indices_within_outer_limit = gas_tree.query_ball_point(subhalopos, rad + delta_r/2)
            indices_within_inner_limit = gas_tree.query_ball_point(subhalopos, rad - delta_r/2)
            shell_indices = np.setdiff1d(indices_within_outer_limit, indices_within_inner_limit)
            if len(shell_indices) == 0:
                masses_hot.append(0)
                masses_cold.append(0)
            else:
                shell_masses = mass0[shell_indices]
                T_shell = T_gas[shell_indices]
                hot_inds = np.where(T_shell > temp_thresh)[0]
                cold_inds = np.where(T_shell <= temp_thresh)[0]
                if hot_inds.size == 0:
                    masses_hot.append(0)
                else:
                    masses_hot.append(np.sum(shell_masses[hot_inds])/delta_r)
                if cold_inds.size == 0:
                    masses_cold.append(0)
                else:
                    masses_cold.append(np.sum(shell_masses[cold_inds])/delta_r)
        return np.array(masses_hot), np.array(masses_cold)
    else:
        masses = []
        for rad, delta_r in zip(rads, delta_rs):
            #find appropriate indices
            indices_within_outer_limit = gas_tree.query_ball_point(subhalopos, rad + delta_r/2)
            indices_within_inner_limit = gas_tree.query_ball_point(subhalopos, rad - delta_r/2)
            shell_indices = np.setdiff1d(indices_within_outer_limit, indices_within_inner_limit)
            if len(shell_indices) == 0:
                shell_mass = 0
            else:
                shell_mass = np.sum(mass0[shell_indices])/delta_r
            masses.append(shell_mass)

        return np.array(masses)