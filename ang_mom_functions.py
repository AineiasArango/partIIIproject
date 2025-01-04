def ang_mom_mag(snap_dir, snap_number, rads, delta_rs):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    import cosmo_utils as cu
    from scipy import spatial
    
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

    ang_mom_mags = []
    for rad, delta_r in zip(rads, delta_rs):
        #find appropriate indices
        indices_within_outer_limit = gas_tree.query_ball_point(subhalopos, rad + delta_r/2)
        indices_within_inner_limit = gas_tree.query_ball_point(subhalopos, rad - delta_r/2)
        shell_indices = np.setdiff1d(indices_within_outer_limit, indices_within_inner_limit)

        unit_conversion = 1 # keeping as km/s
        if shell_indices.size == 0:
            ang_mom_mag = 0
        else:
            #shell quantities   
            v_gas_shell = v_gas0[shell_indices] - central_vel #velocity recentred
            pos_shell = pos0[shell_indices] - subhalopos #particle positions
            mass_shell = mass0[shell_indices]    

            ang_mom_mag = np.sum(mass_shell*np.linalg.norm(np.cross(pos_shell, v_gas_shell), axis=1))/delta_r*unit_conversion #angular momentum per unit distance
        ang_mom_mags.append(ang_mom_mag)
    return np.array(ang_mom_mags)

def ang_mom_mag_T(snap_dir, snap_number, rads, delta_rs, temp_thresh=1e5):
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

    unit_conversion = 1 # keeping as km/s
    ang_mom_mags_hot = []
    ang_mom_mags_cold = []
    for rad, delta_r in zip(rads, delta_rs):
        #find appropriate indices
        indices_within_outer_limit = gas_tree.query_ball_point(subhalopos, rad + delta_r/2)
        indices_within_inner_limit = gas_tree.query_ball_point(subhalopos, rad - delta_r/2)
        shell_indices = np.setdiff1d(indices_within_outer_limit, indices_within_inner_limit)

        if shell_indices.size == 0:
            ang_mom_mag_hot = 0
            ang_mom_mag_cold = 0
        else:
            #shell quantities   
            v_gas_shell = v_gas0[shell_indices] - central_vel #velocity recentred
            pos_shell = pos0[shell_indices] - subhalopos #particle positions
            mass_shell = mass0[shell_indices]
            T_shell = T_gas[shell_indices]

            shell_indices_hot = np.where(T_shell > temp_thresh)[0]
            shell_indices_cold = np.where(T_shell <= temp_thresh)[0]
            ang_mom_mag_hot = np.sum(mass_shell[shell_indices_hot]*np.linalg.norm(np.cross(pos_shell[shell_indices_hot], v_gas_shell[shell_indices_hot]), axis=1)/delta_r)*unit_conversion
            ang_mom_mag_cold = np.sum(mass_shell[shell_indices_cold]*np.linalg.norm(np.cross(pos_shell[shell_indices_cold], v_gas_shell[shell_indices_cold]), axis=1)/delta_r)*unit_conversion
            
        ang_mom_mags_hot.append(ang_mom_mag_hot)
        ang_mom_mags_cold.append(ang_mom_mag_cold)
    return np.array(ang_mom_mags_hot), np.array(ang_mom_mags_cold)

def ang_mom_mag_function(snap_dir, snap_number, rads, delta_rs, split_temp=False, temp_thresh=10**4.25):
    import numpy as np
    if split_temp:
        return ang_mom_mag_T(snap_dir, snap_number, rads, delta_rs, temp_thresh=temp_thresh)
    else:
        return ang_mom_mag(snap_dir, snap_number, rads, delta_rs)

def tot_ang_mom_mag(snap_dir, snap_number, rads):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    import cosmo_utils as cu
    from scipy import spatial
    
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

    ang_mom_mags = []
    for rad in rads:
        #find appropriate indices
        sphere_indices = gas_tree.query_ball_point(subhalopos, rad)

        unit_conversion = 1 # keeping as kpc*km/s
        if sphere_indices.size == 0:
            ang_mom_mag = 0
        else:
            #shell quantities   
            v_gas_shell = v_gas0[sphere_indices] - central_vel #velocity recentred
            pos_shell = pos0[sphere_indices] - subhalopos #particle positions
            mass_shell = mass0[sphere_indices]    

            ang_mom_mag = np.sum(mass_shell*np.linalg.norm(np.cross(pos_shell, v_gas_shell), axis=1))*unit_conversion #angular momentum per unit distance
        ang_mom_mags.append(ang_mom_mag)
    return np.array(ang_mom_mags)

def tot_ang_mom_mag_T(snap_dir, snap_number, rads, temp_thresh=1e5):
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

    unit_conversion = 1 # keeping as kpc*km/s
    ang_mom_mags_hot = []
    ang_mom_mags_cold = []
    
    for rad in rads:
        #find appropriate indices
        sphere_indices = gas_tree.query_ball_point(subhalopos, rad)

        if sphere_indices.size == 0:
            ang_mom_mag_hot = 0
            ang_mom_mag_cold = 0
        else:
            #shell quantities   
            v_gas_shell = v_gas0[sphere_indices] - central_vel #velocity recentred
            pos_shell = pos0[sphere_indices] - subhalopos #particle positions
            mass_shell = mass0[sphere_indices]
            T_shell = T_gas[sphere_indices]

            sphere_indices_hot = np.where(T_shell > temp_thresh)[0]
            sphere_indices_cold = np.where(T_shell <= temp_thresh)[0]
            ang_mom_mag_hot = np.sum(mass_shell[sphere_indices_hot]*np.linalg.norm(np.cross(pos_shell[sphere_indices_hot], v_gas_shell[sphere_indices_hot]), axis=1))*unit_conversion
            ang_mom_mag_cold = np.sum(mass_shell[sphere_indices_cold]*np.linalg.norm(np.cross(pos_shell[sphere_indices_cold], v_gas_shell[sphere_indices_cold]), axis=1))*unit_conversion
            
        ang_mom_mags_hot.append(ang_mom_mag_hot)
        ang_mom_mags_cold.append(ang_mom_mag_cold)
    return np.array(ang_mom_mags_hot), np.array(ang_mom_mags_cold)

def tot_ang_mom_mag_function(snap_dir, snap_number, rads, split_temp=False, temp_thresh=10**4.25):
    import numpy as np
    if split_temp:
        return tot_ang_mom_mag_T(snap_dir, snap_number, rads, temp_thresh=temp_thresh)
    else:
        return tot_ang_mom_mag(snap_dir, snap_number, rads)
    

def ang_mom_mag_vec(snap_dir, snap_number, rads, delta_rs, use_axis=False, axisvec=[1,1,1]):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    import cosmo_utils as cu
    from scipy import spatial
    
    axisvec=np.array(axisvec)
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

    ang_mom_mags = []
    for rad, delta_r in zip(rads, delta_rs):
        #find appropriate indices
        indices_within_outer_limit = gas_tree.query_ball_point(subhalopos, rad + delta_r/2)
        indices_within_inner_limit = gas_tree.query_ball_point(subhalopos, rad - delta_r/2)
        shell_indices = np.setdiff1d(indices_within_outer_limit, indices_within_inner_limit)

        unit_conversion = 1 # keeping as km/s
        if shell_indices.size == 0:
            ang_mom_mag = 0
        else:
            #shell quantities   
            v_gas_shell = v_gas0[shell_indices] - central_vel #velocity recentred
            pos_shell = pos0[shell_indices] - subhalopos #particle positions
            mass_shell = mass0[shell_indices]    

            ang_mom_mag = np.sum(mass_shell*np.cross(pos_shell, v_gas_shell), axis=0)/delta_r*unit_conversion #angular momentum per unit distance
        
            if use_axis:
                ang_mom_mag = np.dot(ang_mom_mag, axisvec)
            else:
                ang_mom_mag = np.linalg.norm(ang_mom_mag)
            ang_mom_mags.append(ang_mom_mag)
    return np.array(ang_mom_mags)

def ang_mom_mag_vec_T(snap_dir, snap_number, rads, delta_rs, temp_thresh=1e5, use_axis=False, axisvec=[1,1,1]):
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
    axisvec=np.array(axisvec)

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

    unit_conversion = 1 # keeping as km/s
    ang_mom_mags_hot = []
    ang_mom_mags_cold = []
    for rad, delta_r in zip(rads, delta_rs):
        #find appropriate indices
        indices_within_outer_limit = gas_tree.query_ball_point(subhalopos, rad + delta_r/2)
        indices_within_inner_limit = gas_tree.query_ball_point(subhalopos, rad - delta_r/2)
        shell_indices = np.setdiff1d(indices_within_outer_limit, indices_within_inner_limit)

        if shell_indices.size == 0:
            ang_mom_mag_hot = 0
            ang_mom_mag_cold = 0
        else:
            #shell quantities   
            v_gas_shell = v_gas0[shell_indices] - central_vel #velocity recentred
            pos_shell = pos0[shell_indices] - subhalopos #particle positions
            mass_shell = mass0[shell_indices]
            T_shell = T_gas[shell_indices]

            shell_indices_hot = np.where(T_shell > temp_thresh)[0]
            shell_indices_cold = np.where(T_shell <= temp_thresh)[0]
            ang_mom_mag_hot = np.sum(mass_shell[shell_indices_hot]*np.cross(pos_shell[shell_indices_hot], v_gas_shell[shell_indices_hot]), axis=0)/delta_r*unit_conversion
            ang_mom_mag_cold = np.sum(mass_shell[shell_indices_cold]*np.cross(pos_shell[shell_indices_cold], v_gas_shell[shell_indices_cold]), axis=0)/delta_r*unit_conversion
            if use_axis:
                ang_mom_mag_hot = np.dot(ang_mom_mag_hot, axisvec)
                ang_mom_mag_cold = np.dot(ang_mom_mag_cold, axisvec)
            else:
                ang_mom_mag_hot = np.linalg.norm(ang_mom_mag_hot)
                ang_mom_mag_cold = np.linalg.norm(ang_mom_mag_cold)
        ang_mom_mags_hot.append(ang_mom_mag_hot)
        ang_mom_mags_cold.append(ang_mom_mag_cold)
    return np.array(ang_mom_mags_hot), np.array(ang_mom_mags_cold)

def ang_mom_mag_vec_function(snap_dir, snap_number, rads, delta_rs, split_temp=False, temp_thresh=10**4.25):
    import numpy as np
    if split_temp:
        return ang_mom_mag_T(snap_dir, snap_number, rads, delta_rs, temp_thresh=temp_thresh)
    else:
        return ang_mom_mag(snap_dir, snap_number, rads, delta_rs)
