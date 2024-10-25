import matplotlib.pyplot as plt
import os
os.chdir("/data/ERCblackholes4/sk939/for_aineias")
import read_fof_files as rff
import read_snap_files as rsf 
import sys
import numpy as np

sys.path.insert(0,"/data/ERCblackholes4/sk939/for_aineias")

snap_dir = "/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"
snap_no = 86

fof_file = rff.get_fof_filename(snap_dir, snap_no)
subhalopos = rff.get_subhalo_data(fof_file, 'SubhaloPos')[0]
Subhalo_Halfmass_rad = rff.get_subhalo_data(fof_file, 'SubhaloHalfmassRadType')[0,0]

snap_name = rsf.get_snap_filename(snap_dir, snap_no)
pos0 = rsf.get_snap_data(snap_name,0,"Coordinates") #use 0 because gas is particle type 0
mass0 = rsf.get_snap_data(snap_name,0,"Masses")

#just subtract subhalopos from every coordinate in pos0

recentred_pos0 = pos0 -subhalopos

# Calculate the distance between each gas cell and the subhalo position
distances = np.sqrt(np.sum((recentred_pos0)**2, axis=1))

outer_limit = 7
bin_width = 0.1
bin_edges = np.arange(0, outer_limit + 1, bin_width)

# Create a list to store the indices for each bin
binned_indices = []

# Sort the distances into bins
for i in range(len(bin_edges) - 1):
    lower_bound = bin_edges[i]
    upper_bound = bin_edges[i+1]
    
    # Get indices of gas cells within the current bin
    bin_indices = np.where((distances >= lower_bound) & (distances < upper_bound))[0]
    
    binned_indices.append(bin_indices)

mass_list = []
for i in binned_indices:
    mass = sum(mass0[i])
    mass_list.append(mass)


# Create a bar chart of the mass distribution
plt.figure(figsize=(12, 6))
plt.bar(bin_edges[:-1], mass_list, width=bin_width, align='edge', edgecolor='black')
plt.xlabel('Distance from Galaxy Centre (ckpc/h)')
plt.ylabel('Total Gas Mass (M$_\odot$)')
plt.title('Gas Mass Distribution in Radial Bins')
plt.xlim(0, outer_limit)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, outer_limit, 1))
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig('gas_mass_distribution.png')
plt.close()



