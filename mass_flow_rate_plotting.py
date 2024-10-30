import matplotlib.pyplot as plt
import numpy as np
import read_fof_files as rff
import read_snap_files as rsf
import mass_flow_rate_function as mfr
import sys
sys.path.insert(0, "/data/ERCblackholes4/sk939/for_aineias")
snap_dir="/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"

data = []

for i in range(45,87):
    snap_number = i
    data.append(mfr.mass_flow_rate(snap_dir, snap_number))

data = np.array(data)
redshifts = data[:,0]
mdot_tot = data[:,1] 
mdot_in = data[:,2]
mdot_out = data[:,3]
beta = data[:,4]

import os
os.chdir("/home/aasnha2/Project")
# Plot mdot_in vs redshift
plt.figure(figsize=(10,6))
plt.plot(redshifts, mdot_in, 'b-', label='Inflow')
plt.xlabel('Redshift')
plt.ylabel('Mass Flow Rate ($10^{10} M_\odot h^{-1} km s^{-1} kpc^{-1}$)')
plt.title('Gas Mass Inflow Rate vs Redshift')
plt.grid(True)
plt.legend()
plt.savefig('mass_inflow_rate.png')
plt.close()

# Plot mdot_out vs redshift
plt.figure(figsize=(10,6))
plt.plot(redshifts, mdot_out, 'r-', label='Outflow')
plt.xlabel('Redshift')
plt.ylabel('Mass Flow Rate ($10^{10} M_\odot h^{-1} km s^{-1} kpc^{-1}$)')
plt.title('Gas Mass Outflow Rate vs Redshift')
plt.grid(True)
plt.legend()
plt.savefig('mass_outflow_rate.png')
plt.close()

# Plot mdot_tot vs redshift
plt.figure(figsize=(10,6))
plt.plot(redshifts, mdot_tot, 'g-', label='Net Flow')
plt.xlabel('Redshift')
plt.ylabel('Mass Flow Rate ($10^{10} M_\odot h^{-1} km s^{-1} kpc^{-1}$)')
plt.title('Net Gas Mass Flow Rate vs Redshift')
plt.grid(True)
plt.legend()
plt.savefig('net_mass_flow_rate.png')
plt.close()

# Plot beta vs redshift
plt.figure(figsize=(10,6))
plt.plot(redshifts, beta, 'k-', label='Beta')
plt.xlabel('Redshift')
plt.ylabel('Mass Loading Factor')
plt.title('Mass Loading Factor vs Redshift')
plt.grid(True)
plt.legend()
plt.savefig('mass_loading_factor.png')
plt.close()