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
    pos0 = rsf.get_snap_data(snap_name,0,"Coordinates") #gas particle positions (kpc)
    gal_pos = cu.cosmo_to_phys(pos0[gal_inds], a, h, length=True) #halo gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (Msun)
    v_gas = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True) #velocity of every gas particle in the snapshot (km/s)    

    #Make the gas tree
    gas_tree = spatial.cKDTree(gal_pos) #only build tree with cells from main halo

    #Find the mass weighted mean velocity of the gas particles within 2*r_vir of the subhalo
    central_inds = gas_tree.query_ball_point(subhalopos, 2.0*r_vir) #indices of gas particles within 2*r_vir of the subhalo
    central_vels = v_gas[central_inds]
    central_masses = mass0[central_inds]
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
            v_gas_shell = v_gas[shell_indices] - central_vel #velocity recentred
            pos_shell = gal_pos[shell_indices] - subhalopos #particle positions
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
    return v_radials