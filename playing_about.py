import numpy as np
def get_smooth_neighbour_variables(snap_dir, snap_number):
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
    subhalopos = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloPos')[0], a, h, length=True) #subhalo position (kpc)
    subhaloCM = cu.cosmo_to_phys(rff.get_subhalo_data(fof_file, 'SubhaloCM')[0], a, h, length=True) #subhalo centre of mass position (kpc)
    pos0 = cu.cosmo_to_phys(rsf.get_snap_data(snap_name,0,"Coordinates"), a, h, length=True) #gas particle positions (kpc)


    gas_tree = spatial.cKDTree(pos0)

    #subhalopos is now at the origin and the perculiar velocity of the subhalo has been taken away.
    central_index = gas_tree.query(subhalopos, k=1)[1]
    central_index_2 = gas_tree.query(subhaloCM, k=1)[1]

    return central_index, central_index_2

snap_dir1 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableLowSNEff"
snap_dir2 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"

for i in [10,50,80]:
    central_index, central_index_2 = get_smooth_neighbour_variables(snap_dir1, i)
    print(central_index, central_index_2)
    central_index, central_index_2 = get_smooth_neighbour_variables(snap_dir2, i)
    print(central_index, central_index_2)
