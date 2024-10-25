
# for the main galaxy/subhalo (i.e. subhalo 0)
import matplotlib.pyplot as plt
import os
os.chdir("/data/ERCblackholes4/sk939/for_aineias")
import read_fof_files as rff
import sys
sys.path.insert(0, "/data/ERCblackholes4/sk939/for_aineias")
snap_dir="/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
redshifts = []
sfrs = []

#loop over each fof file
for i in range(45,87):
    snap_number = i    
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    redshift = rff.get_attribute(fof_file, "Redshift") #redshift
    sfr = rff.get_subhalo_data(fof_file, 'SubhaloSFRinRad')[0] #star formation rate
    redshifts.append(redshift)
    sfrs.append(sfr)

# After the loop, create the plot

plt.figure(figsize=(10, 6))
plt.plot(redshifts, sfrs, 'b-', marker='o')
plt.xlim(1,6)
plt.xlabel('Redshift')
plt.ylabel('Star Formation Rate/(M$_\odot$/yr)')
plt.title('Star Formation Rate vs Redshift')
plt.grid(True)
plt.savefig('sfr_vs_redshift.png')
plt.close()



