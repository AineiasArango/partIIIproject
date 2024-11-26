import mass_functions as mf
import numpy as np
import read_fof_files as rff
import read_snap_files as rsf
from scipy import spatial
import cosmo_utils as cu
import os

snap_dir1 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
snap_dir2 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"
snap_number = 86
os.chdir("/home/aasnha2/Project/Plots/mass_plots")

#Calculate i_max using the virial radius
#We want r to range from r_0 to r_vir
fof_file = rff.get_fof_filename(snap_dir1, snap_number)
a = rff.get_attribute(fof_file, "Time") #scale factor
h = 0.679 #dimensionless Hubble constant
r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)#virial radius (kpc)    
r_0 = 1e-4 #kpc

#Produce rads and delta_rs
#Done so that the bins touch and the bins are 10% the width of the radius
ratio = 0.1
fraction = (1+ratio/2)/(1-ratio/2)
i_max = int(np.log(3*r_vir/r_0)/np.log(fraction)) + 1
rads = r_0*np.logspace(0, i_max, num = i_max + 1, base = fraction)
delta_rs = ratio*rads


#Produce data
#internal mass no temp split
masses1 = mf.mass_function(snap_dir1, snap_number, rads)
masses2 = mf.mass_function(snap_dir2, snap_number, rads)
np.savez('mass_data.npz', masses1=masses1, masses2=masses2, rads=rads)

#internal mass temp split
masses_hot1, masses_cold1 = mf.mass_function(snap_dir1, snap_number, rads, split_temp=True)
masses_hot2, masses_cold2 = mf.mass_function(snap_dir2, snap_number, rads, split_temp=True)
np.savez('mass_data_temp_split.npz', masses_hot1=masses_hot1, masses_cold1=masses_cold1, masses_hot2=masses_hot2, masses_cold2=masses_cold2, rads=rads)

#mass density no temp split
mass_densities1 = mf.mass_density_function(snap_dir1, snap_number, rads, delta_rs)
mass_densities2 = mf.mass_density_function(snap_dir2, snap_number, rads, delta_rs)
np.savez('mass_density_data.npz', mass_densities1=mass_densities1, mass_densities2=mass_densities2, rads=rads)

#mass density temp split
mass_densities_hot1, mass_densities_cold1 = mf.mass_density_function(snap_dir1, snap_number, rads, delta_rs, split_temp=True)
mass_densities_hot2, mass_densities_cold2 = mf.mass_density_function(snap_dir2, snap_number, rads, delta_rs, split_temp=True)
np.savez('mass_density_data_temp_split.npz', mass_densities_hot1=mass_densities_hot1, mass_densities_cold1=mass_densities_cold1, mass_densities_hot2=mass_densities_hot2, mass_densities_cold2=mass_densities_cold2, rads=rads)

#total mass
total_masses1 = mf.total_mass_function(snap_dir1, snap_number, rads, 1536)
total_masses2 = mf.total_mass_function(snap_dir2, snap_number, rads, 192)
np.savez('total_mass_data.npz', total_masses1=total_masses1, total_masses2=total_masses2, rads=rads)