import numpy as np
#function to decide what to use as the sample radius. This is the radius of the sphere that contains 50 nearest neighbours
def get_sample_radius(subhalopos, pos0, n_neighbours=50):
    import scipy.spatial as spatial
    gas_tree = spatial.cKDTree(pos0)
    neighbour_distances = gas_tree.query(subhalopos, k=n_neighbours)[0]
    return max(neighbour_distances)

#Find the cells that are adjacent to a given cell
def find_touching_cells(vor, cell_index):
    # Find all ridges involving our cell of interest
    touching_cells = []
    for p1, p2 in vor.ridge_points:
        if p1 == cell_index:
            touching_cells.append(p2)
        elif p2 == cell_index:
            touching_cells.append(p1)
    
    return touching_cells

#Find the central cell of a subhalo
def find_central_pos(subhalopos, pos0):
    import numpy as np
    import read_fof_files as rff
    import read_snap_files as rsf
    import cosmo_utils as cu
    from scipy import spatial
    sample_radius = get_sample_radius(subhalopos, pos0)
    gas_tree = spatial.cKDTree(pos0)
    nearest_neighbour_inds = gas_tree.query_ball_point(subhalopos, sample_radius)
    nearest_neighbour_pos = pos0[nearest_neighbour_inds]
    pos_inds = np.arange(len(nearest_neighbour_inds))

    # Create dictionary mapping indices in pos_inds to indices in nearest_neighbour_inds, and hence the gas tree
    pos_index_to_true_index = {pos: idx for pos, idx in zip(pos_inds, nearest_neighbour_inds)}
    #Find the index of the origin
    origin_index = np.argmin(np.linalg.norm(nearest_neighbour_pos-subhalopos, axis=1))

    central_pos = pos0[pos_index_to_true_index[origin_index]] 
    return central_pos, pos_index_to_true_index[origin_index]


#Turn this all into a function. This function gets the properties of the gas particle adjacent to the cetnral particle.
def get_neighbour_variables(snap_dir, snap_number):
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

    #get host group
    host_groups = cu.get_host_groups(snap_dir, snap_number, 0)
    gal_inds = host_groups == 0
    
    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    redshift = rff.get_attribute(fof_file, "Redshift") #redshift
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)#virial radius (kpc)    
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)
    gal_pos = pos0[gal_inds] #halo gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (Msun)
    gal_mass = mass0[gal_inds]
    v_gas0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True) #velocity of every gas particle in the snapshot (km/s)    
    gal_v_gas = v_gas0[gal_inds]
    density0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Density"), a, h, density=True) #density of every gas particle in the snapshot (Msun/kpc^3)
    x_e = rsf.get_snap_data(snap_name,0,"ElectronAbundance") #electron abundance, n_e/n_H
    int_energy = rsf.get_snap_data(snap_name,0,"InternalEnergy")*(10**10) #internal energy of every gas particle in the snapshot (cm/s)^2

    #Make the gas tree
    gal_gas_tree = spatial.cKDTree(gal_pos) #only build tree with cells from main halo
    gas_tree = spatial.cKDTree(pos0)

    #Find the mass weighted mean velocity of the gas particles within 2*r_vir of the subhalo
    central_inds = gal_gas_tree.query_ball_point(subhalopos, 2.0*r_vir) #indices of gas particles within 2*r_vir of the subhalo
    central_vels = gal_v_gas[central_inds]
    central_masses = gal_mass[central_inds]
    central_vel = np.average(central_vels, axis=0, weights=central_masses) #mass weighted mean velocity
    v_gas0 = v_gas0 - central_vel

    #subhalopos is now at the origin and the perculiar velocity of the subhalo has been taken away.

    #find the nearest neighbours of the origin
    sample_radius = get_sample_radius(subhalopos, pos0)
    neighbour_inds = gas_tree.query_ball_point(subhalopos, sample_radius)
    neighbour_pos = pos0[neighbour_inds]
    pos_inds = np.arange(len(neighbour_inds))

    # Create dictionary mapping indices in pos_inds to indices in neighbour_inds, and hence the gas tree
    pos_index_to_true_index = {pos: idx for pos, idx in zip(pos_inds, neighbour_inds)}
    #Find the index of the origin
    origin_index = np.argmin(np.linalg.norm(neighbour_pos-subhalopos, axis=1))
    #Now create the voronoi tesselation with the particles in neighbour_pos
    vor = Voronoi(neighbour_pos)
    #Find cells adjacent to origin cell
    touching_cells = find_touching_cells(vor, origin_index)
    #Find the indices of these cells in the actual gas_tree so you can work out variables at these points
    #Do this by finding the indices of the touching cells in the pos_index_to_true_index dictionary
    touching_cells_indices = [pos_index_to_true_index[cell] for cell in touching_cells]

    #Now you can work out variables at these points (position, mass, density, temperature, velocity)
    n_neighbours = len(touching_cells_indices)
    central_pos = pos0[pos_index_to_true_index[origin_index]]-subhalopos #This is the cell coordinate of the central cell, so it won't be at exactly [0,0,0]
    adj_pos = pos0[touching_cells_indices]-subhalopos
    adj_masses = mass0[touching_cells_indices]
    adj_densities = density0[touching_cells_indices]
    adj_vels = v_gas0[touching_cells_indices]
    mu = 4*m_proton/(1+3*X_H+4*X_H*x_e[touching_cells_indices])
    adj_Ts = (gamma-1)*int_energy[touching_cells_indices]*mu/k_B #temperature of gas cells (K)

    return central_pos, adj_pos, adj_masses, adj_densities, adj_vels, adj_Ts, n_neighbours, r_vir, redshift

def get_smooth_neighbour_variables(snap_dir, snap_number, num_neighbours=32, subhalopos=True, subhalocm=False, central_coord=np.array([0,0,0])):
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

    #get host group
    host_groups = cu.get_host_groups(snap_dir, snap_number, 0)
    gal_inds = host_groups == 0
    
    #get attributes and convert units
    a = rff.get_attribute(fof_file, "Time") #scale factor
    redshift = rff.get_attribute(fof_file, "Redshift") #redshift
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    subhaloCM = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloCM')[0], a, h, length=True) #subhalo centre of mass position (kpc)
    central_coord = cu.cosmo_to_phys(central_coord, a, h, length=True)
    r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)#virial radius (kpc)    
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)
    gal_pos = pos0[gal_inds] #halo gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (Msun)
    gal_mass = mass0[gal_inds]
    v_gas0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True) #velocity of every gas particle in the snapshot (km/s)    
    gal_v_gas = v_gas0[gal_inds]
    density0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Density"), a, h, density=True) #density of every gas particle in the snapshot (Msun/kpc^3)
    x_e = rsf.get_snap_data(snap_name,0,"ElectronAbundance") #electron abundance, n_e/n_H
    int_energy = rsf.get_snap_data(snap_name,0,"InternalEnergy")*(10**10) #internal energy of every gas particle in the snapshot (cm/s)^2

    #Make the gas tree
    gal_gas_tree = spatial.cKDTree(gal_pos) #only build tree with cells from main halo
    gas_tree = spatial.cKDTree(pos0)

    if subhalopos:
        subhalopos = subhalopos
    elif subhalocm:
        subhalopos = subhalocm
    else:
        subhalopos = central_coord
    #Find the mass weighted mean velocity of the gas particles within 2*r_vir of the subhalo
    central_inds = gal_gas_tree.query_ball_point(subhalopos, 2.0*r_vir) #indices of gas particles within 2*r_vir of the subhalo
    central_vels = gal_v_gas[central_inds]
    central_masses = gal_mass[central_inds]
    central_vel = np.average(central_vels, axis=0, weights=central_masses) #mass weighted mean velocity


    #subhalopos is now at the origin and the perculiar velocity of the subhalo has been taken away.
    central_index = gas_tree.query(subhalopos, k=1)[1]
    central_pos = pos0[central_index]
    neighbour_inds = gas_tree.query(central_pos, k=num_neighbours+1)[1]
    # Remove central index from neighbour indices
    neighbour_inds = [idx for idx in neighbour_inds if idx != central_index]
    neighbour_pos = pos0[neighbour_inds] - central_pos
    neighbour_v_gas = v_gas0[neighbour_inds] - central_vel
    neighbour_densities = density0[neighbour_inds]
    mu = 4*m_proton/(1+3*X_H+4*X_H*x_e[neighbour_inds])
    neighbour_Ts = (gamma-1)*int_energy[neighbour_inds]*mu/k_B #temperature of gas cells (K)   
    neighbour_masses = mass0[neighbour_inds]
    neighbour_data = np.column_stack((neighbour_pos, neighbour_v_gas, neighbour_densities, neighbour_Ts, neighbour_masses))

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
def in_or_out(snap_dir, snap_number, num_neighbours=32, subhalopos=True, subhalocm=False, central_coord=np.array([0,0,0])):
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
    subhalocm = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloCM')[0], a, h, length=True) #subhalo centre of mass position (kpc)
    central_coord = cu.cosmo_to_phys(central_coord, a, h, length=True)
    r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)#virial radius (kpc)
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)
    gal_pos = pos0[gal_inds] #halo gas particle positions (kpc)
    mass0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Masses"), a, h, mass=True) #gas particle masses (Msun)
    gal_mass = mass0[gal_inds]
    density0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Density"), a, h, density=True) #density of every gas particle in the snapshot (Msun/kpc^3)
    v_gas0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Velocities"), a, h, velocity=True) #velocity of every gas particle in the snapshot (km/s)    
    gal_v_gas = v_gas0[gal_inds]

    #Make the gas tree
    gal_gas_tree = spatial.cKDTree(gal_pos) #only build tree with cells from main halo
    gas_tree = spatial.cKDTree(pos0)

    #decide which position to use as a central cell
    if subhalopos:
        subhalopos = subhalopos
    elif subhalocm:
        subhalopos = subhalocm
    else:
        subhalopos = central_coord
    #Find the mass weighted mean velocity of the gas particles within 2*r_vir of the subhalo
    central_inds = gal_gas_tree.query_ball_point(subhalopos, 2.0*r_vir) #indices of gas particles within 2*r_vir of the subhalo
    central_vels = gal_v_gas[central_inds]
    central_masses = gal_mass[central_inds]
    central_vel = np.average(central_vels, axis=0, weights=central_masses) #mass weighted mean velocity

    central_index = gas_tree.query(subhalopos, k=1)[1]
    central_pos = pos0[central_index]
    neighbour_distances, neighbour_inds = gas_tree.query(central_pos, k=num_neighbours+1)
    smoothing_length = max(neighbour_distances)
    # Remove central index from neighbour indices
    neighbour_inds = [idx for idx in neighbour_inds if idx != central_index]
    neighbour_v_gas = v_gas0[neighbour_inds] - central_vel

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

def in_or_out_2(neighbour_pos, neighbour_v_gas, neighbour_densities):
    import numpy as np

    norms = np.linalg.norm(neighbour_pos, axis=1)[:, np.newaxis]    
    v_radial = np.einsum('ij,ij->i', neighbour_v_gas, neighbour_pos/norms) #radial component of velocity from dot product

    smoothing_length = max(norms)
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


"""
#Data gathering
import numpy as np
import os
os.chdir("/home/aasnha2/Project/Plots/Neighbour_variable_plots")
snap_dir1 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
snap_dir2 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"

for snapdir, name in [(snap_dir1, "NoBHFableHighSNEff"), (snap_dir2, "NoBHFableHighSNEffHighRes")]:
    positions = []
    masses = []
    densities = []
    velocities = []
    temps = []
    n_neighbours_list = []
    r_vir_list = []
    redshifts_list = []
    for i in range(54,87):  
        central_pos, adj_pos, adj_masses, adj_densities, adj_vels, adj_Ts, n_neighbours, r_vir, redshift = get_neighbour_variables(snapdir, i)
        positions.extend(adj_pos)
        masses.extend(adj_masses)
        densities.extend(adj_densities)
        velocities.extend(adj_vels)
        temps.extend(adj_Ts)
        n_neighbours_list.append(n_neighbours)
        r_vir_list.append(r_vir)
        redshifts_list.append(redshift)

    np.savez('neighbour_variables_dg_'+name+'.npz', positions=positions, masses=masses, densities=densities, velocities=velocities, temps=temps, n_neighbours=n_neighbours_list, r_vir=r_vir_list, redshifts=redshifts_list)
"""

if __name__ == '__main__':

    snap_dir1 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableLowSNEff"
    snap_dir2 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableLowSNEffHighRes"

    # Initialize arrays to store results
    smooth_results = []
    inout_results = []

    # Loop over snapshots
    for i in range(40, 87):
        try:
            # Get results from both functions for this snapshot
            smooth_result = get_smooth_neighbour_variables(snap_dir1, i)
            inout_result = in_or_out(snap_dir2, i)
            
            # Append results to arrays
            smooth_results.append(smooth_result)
            inout_results.append(inout_result)
            
        except Exception as e:
            print(f"Error processing snapshot {i}: {e}")
            continue

    # Convert to numpy arrays
    smooth_results = np.array(smooth_results)
    inout_results = np.array(inout_results)

    import os
    os.chdir("/data/ERCblackholes4/aasnha2/for_aineias/plots")
    # Save results to npz file
    np.savez('smooth_and_inout_test_results_lowSNEff.npz', 
            smooth_results=smooth_results,
            inout_results=inout_results)




