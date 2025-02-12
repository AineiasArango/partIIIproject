import numpy as np
import numpy as np
import read_fof_files as rff
import read_snap_files as rsf
import cosmo_utils as cu
from scipy import spatial
from scipy.spatial import Voronoi
from scipy import constants

def get_smooth_neighbour_variables(snap_dir, snap_number, num_neighbours=32):
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
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    subhalovel = rff.get_subhalo_data(fof_file, 'SubhaloVel')[0]
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (Msun)
    v_gas0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True) #velocity of every gas particle in the snapshot (km/s)    
    density0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Density"), a, h, density=True) #density of every gas particle in the snapshot (Msun/kpc^3)
    x_e = rsf.get_snap_data(snap_name,0,"ElectronAbundance") #electron abundance, n_e/n_H
    int_energy = rsf.get_snap_data(snap_name,0,"InternalEnergy")*(10**10) #internal energy of every gas particle in the snapshot (cm/s)^2

    #Make the gas tree
    gas_tree = spatial.cKDTree(pos0)
    #find the central cell
    central_index = gas_tree.query(subhalopos, k=1)[1]
    central_pos = pos0[central_index]
    #find the neighbours
    neighbour_inds = gas_tree.query(central_pos, k=num_neighbours+1)[1]
    # Remove central index from neighbour indices
    neighbour_inds = [idx for idx in neighbour_inds if idx != central_index]
    #find the properties of each neighbour
    neighbour_pos = pos0[neighbour_inds] - central_pos
    radial_pos = np.linalg.norm(neighbour_pos, axis=1)
    neighbour_v_gas = v_gas0[neighbour_inds] - subhalovel
    radial_v_gas = np.einsum('ij,ij->i', neighbour_v_gas, neighbour_pos/radial_pos[:,np.newaxis])
    neighbour_densities = density0[neighbour_inds]
    mu = 4*m_proton/(1+3*X_H+4*X_H*x_e[neighbour_inds])
    neighbour_Ts = (gamma-1)*int_energy[neighbour_inds]*mu/k_B #temperature of gas cells (K)   
    neighbour_masses = mass0[neighbour_inds]
    #stack the properties into an array
    neighbour_data = np.column_stack((radial_pos, radial_v_gas, neighbour_densities, neighbour_masses, neighbour_Ts))

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
def in_or_out(snap_dir, snap_number, num_neighbours=32):
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

    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    subhalocm = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloCM')[0], a, h, length=True) #subhalo centre of mass position (kpc)
    subhalovel = rff.get_subhalo_data(fof_file, 'SubhaloVel')[0]
    r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)#virial radius (kpc)
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (Msun)
    density0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Density"), a, h, density=True) #density of every gas particle in the snapshot (Msun/kpc^3)
    v_gas0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True) #velocity of every gas particle in the snapshot (km/s)    

    #Make the gas tree
    gas_tree = spatial.cKDTree(pos0)


    central_index = gas_tree.query(subhalopos, k=1)[1]
    central_pos = pos0[central_index]
    neighbour_distances, neighbour_inds = gas_tree.query(central_pos, k=num_neighbours+1)
    smoothing_length = max(neighbour_distances)
    # Remove central index from neighbour indices
    neighbour_inds = [idx for idx in neighbour_inds if idx != central_index]
    neighbour_v_gas = v_gas0[neighbour_inds] - subhalovel

    #Calculate radial velocities by taking dot product of velocity with normalized position vector
    neighbour_pos = pos0[neighbour_inds] - central_pos
    radial_pos = np.linalg.norm(neighbour_pos, axis=1)[:, np.newaxis]
    v_radial = np.einsum('ij,ij->i', neighbour_v_gas, neighbour_pos/radial_pos) #radial component of velocity from dot product
    neighbour_densities = density0[neighbour_inds]

    # Pre-calculate normalised distances
    r = np.linalg.norm(neighbour_pos, axis=1)
    # Calculate kernel weights
    weights = W_func(r, smoothing_length)
    # Calculate mass flux
    mass_flux = np.sum(neighbour_densities * weights * v_radial) / np.sum(weights)

    if mass_flux < 0:
        return 1
    else:
        return 0

def get_virial_neighbours(snap_dir, snap_number):
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

    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)#virial radius (kpc)
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)   

    #Make the gas tree
    gas_tree = spatial.cKDTree(pos0)
    num_neighbours = len(gas_tree.query_ball_point(subhalopos, r_vir))
    return num_neighbours


if __name__ == "__main__":
    snap_dir1 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableLowSNEff"
    snap_dir2 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableLowSNEffHighRes"
    snap_dir3 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
    snap_dir4 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"

    for snap_dir in [snap_dir1, snap_dir3]:
        for snap_number in range(8,87):
            virial_neighbours = get_virial_neighbours(snap_dir, snap_number)
            if virial_neighbours == 0:
                print(f"No virial neighbours for snapshot {snap_number}")
            else:
                log2_max = int(np.log2(virial_neighbours))
                if log2_max < 4:
                    print(f"Virial radius is too small for snapshot {snap_number}")
                else:
                    for num_neighbours in [2**i for i in range(4, log2_max+1)]:
                        neighbour_data = get_smooth_neighbour_variables(snap_dir, snap_number, num_neighbours=num_neighbours)
                        import os
                        os.chdir("/data/ERCblackholes4/aasnha2/for_aineias/plots")
                        np.savez(f'smooth_results_{os.path.basename(snap_dir)}_{snap_number}_{num_neighbours}.npz', 
                                neighbour_data=neighbour_data)
                        print(f"Saved smooth results for snapshot {snap_number} with {num_neighbours} neighbours")



