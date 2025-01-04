import numpy as np
import matplotlib.pyplot as plt
import cosmo_utils as cu
import os
import read_fof_files as rff
from scipy import constants
from plotting_function import radial_plotting_function

snap_number = 86
os.chdir("/home/aasnha2/Project/Plots/ang_mom_plots")
save_dir = "/home/aasnha2/Project/Plots/ang_mom_plots"
r_vir_label = False

#get the virial radius of the particular snapshot
snap_dir1 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
snap_dir2 = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"
fof_file = rff.get_fof_filename(snap_dir2, snap_number)
a = rff.get_attribute(fof_file, "Time") #scale factor
h = 0.679 #dimensionless Hubble constant
r_vir = cu.cosmo_to_phys(rff.get_group_data(fof_file, "Group_R_Crit200")[0], a, h, length=True)*1000 #virial radius (pc)    
Msun = 1.988416e30 #kg
Redshift = rff.get_attribute(fof_file, "Redshift")
print(Redshift)
r_soft1 = cu.cosmo_to_phys(0.387, a, h, length=True)*1000 #softening length (pc)
r_soft2 = cu.cosmo_to_phys(0.194, a, h, length=True)*1000 #softening length (pc)

data = np.load('ang_mom_data.npz')
ang_mom_mags_hot1 = data['ang_mom_mags_hot1']
ang_mom_mags_cold1 = data['ang_mom_mags_cold1']
ang_mom_mags_hot2 = data['ang_mom_mags_hot2']
ang_mom_mags_cold2 = data['ang_mom_mags_cold2']
rads = data['rads']*1000 #convert to pc

radial_plotting_function(ydata=[ang_mom_mags_hot1, ang_mom_mags_hot2, ang_mom_mags_cold1, ang_mom_mags_cold2], xdata=rads, data_labels=['NoAGNHighSN (hot)', 'NoAGNHighSNRes (hot)', 'NoAGNHighSN (cold)', 'NoAGNHighSNRes (cold)'], 
                         xlabel='r [pc]', ylabel=r'$|\mathrm{J}| [\mathrm{M}_\odot \mathrm{km/s}]$', temp_split=True, r_vir=r_vir, save_name='ang_mom_mag_plt.png', save_dir=save_dir, r_soft1=r_soft1, r_soft2=r_soft2, r_vir_label=r_vir_label)

