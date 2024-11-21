def mass_function(snap_dir, snap_number, rads):
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

    #get host group
    host_groups = cu.get_host_groups(snap_dir, snap_number, 0)
    gal_inds = host_groups == 0

    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)  
    pos0 = rsf.get_snap_data(snap_name,0,"Coordinates") #gas particle positions (kpc)
    gal_pos = cu.cosmo_to_phys(pos0[gal_inds], a, h, length=True) #halo gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (kg)

    #Make the gas tree
    gas_tree = spatial.cKDTree(gal_pos) #only build tree with cells from main halo

    masses = [] 
    for rad in rads:
        indices_within_r = gas_tree.query_ball_point(subhalopos, rad)
        masses_within_r = mass0[indices_within_r]
        mass_within_r = np.sum(masses_within_r)
        masses.append(mass_within_r)

    return np.array(masses)