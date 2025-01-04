import numpy as np
import read_fof_files as rff
import read_snap_files as rsf
import cosmo_utils as cu
from scipy import spatial
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy import constants

#constants
h = 0.679 #dimensionless Hubble constant
k_B = constants.k*1e7 #Boltzmann constant (erg/K)
m_proton = constants.m_p*1e3 #proton mass (g)
X_H = 0.76 #hydrogen mass fraction
gamma = 5/3
    #get fof file and snap file

snap_dir1 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
snap_dir2 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"
snap_number = 54
fof_file = rff.get_fof_filename(snap_dir2, snap_number)
snap_name = rsf.get_snap_filename(snap_dir2, snap_number)

#get host group
host_groups = cu.get_host_groups(snap_dir2, snap_number, 0)
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

#function to decide what to use as the sample radius. This is the radius of the sphere that contains 50 nearest neighbours
def get_sample_radius(subhalopos, pos0, n_neighbours=50):
    gas_tree = spatial.cKDTree(pos0)
    neighbour_distances = gas_tree.query(subhalopos, k=n_neighbours)[0]
    return max(neighbour_distances)

sample_radius = get_sample_radius(subhalopos, pos0)
#find the nearest neighbours of the origin
neighbour_inds = gas_tree.query_ball_point(subhalopos, sample_radius)
neighbour_pos = pos0[neighbour_inds]
pos_inds = np.arange(len(neighbour_inds))



plt.plot(np.linalg.norm(neighbour_pos-subhalopos, axis=1), neighbour_inds, 'o', linestyle='')

def find_touching_cells(vor, cell_index):
    # Find all ridges involving our cell of interest
    touching_cells = []
    for p1, p2 in vor.ridge_points:
        if p1 == cell_index:
            touching_cells.append(p2)
        elif p2 == cell_index:
            touching_cells.append(p1)
    
    return touching_cells
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
central_pos = pos0[pos_index_to_true_index[origin_index]]-subhalopos #This is the cell coordinate of the central cell, so it won't be at exactly the subhalopos
adj_pos = pos0[touching_cells_indices]-subhalopos
adj_masses = mass0[touching_cells_indices]
adj_densities = density0[touching_cells_indices]
adj_vels = v_gas0[touching_cells_indices]
mu = 4*m_proton/(1+3*X_H+4*X_H*x_e[touching_cells_indices])
adj_T = (gamma-1)*int_energy[touching_cells_indices]*mu/k_B #temperature of gas cells (K)

print(adj_pos)