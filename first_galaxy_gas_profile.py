import matplotlib.pyplot as plt
import os
import read_fof_files as rff
import read_snap_files as rsf 
import sys
import numpy as np
from scipy import stats

os.chdir("/data/ERCblackholes4/sk939/for_aineias")
sys.path.insert(0,"/data/ERCblackholes4/sk939/for_aineias")

snap_dir = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"
snap_no = 86

#get files
fof_file = rff.get_fof_filename(snap_dir, snap_no) #fof file
snap_name = rsf.get_snap_filename(snap_dir, snap_no) #snap file

#get attributes
subhalopos = rff.get_subhalo_data(fof_file, 'SubhaloPos')[0] #subhalo position
Subhalo_Halfmass_rad = rff.get_subhalo_data(fof_file, 'SubhaloHalfmassRadType')[0,0] #subhalo half mass radius for gas
pos0 = rsf.get_snap_data(snap_name,0,"Coordinates") #coordinates of gas (0) particles
mass0 = rsf.get_snap_data(snap_name,0,"Masses") #masses of gas (0) particles

#recentre coordinates
recentred_pos0 = pos0 -subhalopos

# distance of each gas particle from the subhalo centre
distances = np.sqrt(np.sum((recentred_pos0)**2, axis=1))

#binning data for histogram
bin_width = 0.1
outer_limit = 10
bin_values, bin_edges, binnumber = stats.binned_statistic(distances, mass0, statistic='sum', bins=np.arange(0, outer_limit + bin_width, bin_width))

# Create a bar chart of the mass distribution
plt.figure(figsize=(12, 6))
plt.bar(bin_edges[:-1], bin_values, width=bin_width, align='edge', edgecolor='black')
plt.xlabel('Distance from Galaxy Centre (ckpc/h)')
plt.ylabel('Total Gas Mass (M$_\odot$)')
plt.title('Gas Mass Distribution in Radial Bins')
plt.xlim(0, outer_limit)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, outer_limit, 1))
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.tight_layout()
os.chdir("/home/aasnha2/Project/Plots")
plt.savefig('gas_mass_distribution.png')
plt.close()



