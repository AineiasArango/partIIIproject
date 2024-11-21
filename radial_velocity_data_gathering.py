import numpy as np
import read_fof_files as rff
import read_snap_files as rsf
from scipy import spatial
import cosmo_utils as cu
import radial_velocity_function as rvf
import os

snap_dir1="/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
snap_dir2="/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"
snap_number = 86

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

v_radials1 = np.array(rvf.radial_velocities(snap_dir1, snap_number, rads, delta_rs))
v_radials2 = np.array(rvf.radial_velocities(snap_dir2, snap_number, rads, delta_rs))
os.chdir("/home/aasnha2/Project/Plots")
np.savez('radial_velocity_data.npz', v_radials1=v_radials1, v_radials2=v_radials2, rads=rads)
