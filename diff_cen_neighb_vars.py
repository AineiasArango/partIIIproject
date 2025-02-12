import numpy as np
def get_smooth_neighbour_variables(snap_dir, snap_number, central_coord, num_neighbours=32):
    #returns array of position, velocity, density, temperature, and mass of gas particles within smoothing_length of the subhalo
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    import cosmo_utils as cu
    from scipy import spatial
    from scipy.spatial import Voronoi
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
    redshift = rff.get_attribute(fof_file, "Redshift") #redshift

    r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)#virial radius (kpc)    
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (Msun)
    v_gas0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True) #velocity of every gas particle in the snapshot (km/s)  
    density0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Density"), a, h, density=True) #density of every gas particle in the snapshot (Msun/kpc^3)
    x_e = rsf.get_snap_data(snap_name,0,"ElectronAbundance") #electron abundance, n_e/n_H
    int_energy = rsf.get_snap_data(snap_name,0,"InternalEnergy")*(10**10) #internal energy of every gas particle in the snapshot (cm/s)^2
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    subhaloCM = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloCM')[0], a, h, length=True) #subhalo centre of mass position (kpc)
    #Make the gas tree

    gas_tree = spatial.cKDTree(pos0)
    #subhalopos is now at the origin and the perculiar velocity of the subhalo has been taken away.
    central_index = gas_tree.query(central_coord, k=1)[1]
    central_pos = pos0[central_index]
    neighbour_inds = gas_tree.query(central_pos, k=num_neighbours+1)[1]
    # Remove central index from neighbour indices
    neighbour_inds = [idx for idx in neighbour_inds if idx != central_index]
    neighbour_pos = pos0[neighbour_inds] - central_pos
    radial_pos = np.linalg.norm(neighbour_pos, axis=1)[:, np.newaxis]
    neighbour_v_gas = v_gas0[neighbour_inds] - v_gas0[central_index]
    radial_v_gas = np.einsum('ij,ij->i', neighbour_v_gas, neighbour_pos/radial_pos)
    neighbour_densities = density0[neighbour_inds]
    mu = 4*m_proton/(1+3*X_H+4*X_H*x_e[neighbour_inds])
    neighbour_Ts = (gamma-1)*int_energy[neighbour_inds]*mu/k_B #temperature of gas cells (K)   
    neighbour_masses = mass0[neighbour_inds]
    neighbour_data = np.column_stack((radial_pos, radial_v_gas, neighbour_densities, neighbour_Ts, neighbour_masses))

    return neighbour_data

#W function for calculating mass flux
def W_func(r, smoothing_length):
    import numpy as np
    x = r/smoothing_length
    result = np.zeros_like(x)
    
    # First condition: 0 <= r/h <= 1/2
    mask1 = (x <= 1/2) & (x >= 0)
    result[mask1] = (1 - 6*x[mask1]**2 + 6*x[mask1]**3) * 8/(np.pi*smoothing_length**3)
    
    # Second condition: 1/2 < r/h <= 1
    mask2 = (x > 1/2) & (x <= 1)
    result[mask2] = 2*(1-x[mask2])**3 * 8/(np.pi*smoothing_length**3)
    
    return result

#categroize the mass flux as inflowing or outflowing
def in_or_out(snap_dir, snap_number, central_coord, num_neighbours=32):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    from scipy import spatial
    from scipy import constants
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
    subhalocm = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloCM')[0], a, h, length=True) #subhalo centre of mass position (kpc)
    central_coord = cu.cosmo_to_phys(central_coord, a, h, length=True)
    r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)#virial radius (kpc)
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (Msun)
    density0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Density"), a, h, density=True) #density of every gas particle in the snapshot (Msun/kpc^3)
    v_gas0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True) #velocity of every gas particle in the snapshot (km/s)    


    #Make the gas tree
    gas_tree = spatial.cKDTree(pos0)

    central_index = gas_tree.query(central_coord, k=1)[1]
    central_pos = pos0[central_index]
    neighbour_distances, neighbour_inds = gas_tree.query(central_pos, k=num_neighbours+1)
    smoothing_length = max(neighbour_distances)
    # Remove central index from neighbour indices
    neighbour_inds = [idx for idx in neighbour_inds if idx != central_index]
    neighbour_v_gas = v_gas0[neighbour_inds] - v_gas0[central_index]

    #Calculate radial velocities by taking dot product of velocity with normalized position vector
    neighbour_pos = pos0[neighbour_inds] - central_pos
    norms = np.linalg.norm(neighbour_pos, axis=1)[:, np.newaxis]    
    v_radial = np.einsum('ij,ij->i', neighbour_v_gas, neighbour_pos/norms) #radial component of velocity from dot product
    neighbour_densities = density0[neighbour_inds]

    # Pre-calculate normalised distances
    r = np.array([np.linalg.norm(pos) for pos in neighbour_pos])
    # Calculate kernel weights
    weights = W_func(r, smoothing_length)
    # Calculate mass flux
    mass_flux = np.sum(neighbour_densities * weights * v_radial) / np.sum(weights)

    if mass_flux < 0:
        return 1
    else:
        return 0


def get_neighbour_pos(snap_dir, snap_number, num_neighbours=32, use_subhalopos=True):
        #returns array of position, velocity, density, temperature, and mass of gas particles within smoothing_length of the subhalo
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    import cosmo_utils as cu
    from scipy import spatial
    from scipy.spatial import Voronoi
    from scipy import constants

    #constants
    h = 0.679 #dimensionless Hubble constant

    #get fof file and snap file
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    snap_name = rsf.get_snap_filename(snap_dir, snap_number)
    
    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    subhalocm = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloCM')[0], a, h, length=True) #subhalo centre of mass position (kpc)
    if use_subhalopos:
        central_coord = subhalopos
    else:
        central_coord = subhalocm
    #Make the gas tree
    gas_tree = spatial.cKDTree(pos0)
    #subhalopos is now at the origin and the perculiar velocity of the subhalo has been taken away.
    central_index = gas_tree.query(central_coord, k=1)[1]
    central_pos = pos0[central_index]
    neighbour_inds = gas_tree.query(central_pos, k=num_neighbours+1)[1]
    neighbour_pos = pos0[neighbour_inds]
    return neighbour_pos

import os
if __name__ == '__main__':

    snap_dir1 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableLowSNEff"
    snap_dir2 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableLowSNEffHighRes"
    snap_dir3 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
    snap_dir4 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"

    # Initialize arrays to store results

    for lowres,highres in [(snap_dir1, snap_dir2), (snap_dir3, snap_dir4)]:
        neighbour_variables_list = []
        result_list = []
        for use_subhalopos in [True, False]:
            for snap_number in range(10, 87):
                neighbour_pos = get_neighbour_pos(lowres, snap_number, num_neighbours=20, use_subhalopos=use_subhalopos)
                for i, central_coord in enumerate(neighbour_pos):
                    print(f"\rProcessing neighbour {i+1}/{len(neighbour_pos)} for snap_number {snap_number} and use_subhalopos {use_subhalopos}", end="")
                    neighbour_variables = get_smooth_neighbour_variables(lowres, snap_number, central_coord)
                    result = in_or_out(highres, snap_number, central_coord)
                    neighbour_variables_list.append(neighbour_variables)
                    result_list.append(result)
                print() # New line after progress bar
        neighbour_variables_array = np.array(neighbour_variables_list)
        result_array = np.array(result_list)
        os.chdir("/data/ERCblackholes4/aasnha2/for_aineias/plots")
        np.savez(f'smooth_inout_results_{os.path.basename(lowres)}_diff_cen_neighb.npz', 
            smooth_results=neighbour_variables_array,
            inout_results=result_array)

    
