import matplotlib.pyplot as plt
import numpy as np
import read_fof_files as rff
import read_snap_files as rsf
import mass_flow_rate_function as mfr
import sys
from astropy.cosmology import Planck18
sys.path.insert(0, "/data/ERCblackholes4/sk939/for_aineias")
snap_dir1="/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"

data1 = []

for i in range(45,87):
    snap_number = i
    data1.append(mfr.mass_flow_rate(snap_dir1, snap_number))

data1 = np.array(data1)
redshifts1 = data1[:,0]
mdot_tot1 = data1[:,1] 
mdot_in1 = data1[:,2]
mdot_out1 = data1[:,3]
beta1 = data1[:,4]

snap_dir2="/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"
data2 = []
for i in range(45,87):
    snap_number = i
    data2.append(mfr.mass_flow_rate(snap_dir2, snap_number))

data2 = np.array(data2)
redshifts2 = data2[:,0]
mdot_tot2 = data2[:,1] 
mdot_in2 = data2[:,2]
mdot_out2 = data2[:,3]
beta2 = data2[:,4]
import os
os.chdir("/home/aasnha2/Project/Plots")

np.savez('mass_flow_rate_data.npz', redshifts1=redshifts1, mdot_tot1=mdot_tot1, mdot_in1=mdot_in1, mdot_out1=mdot_out1, beta1=beta1, redshifts2=redshifts2, mdot_tot2=mdot_tot2, mdot_in2=mdot_in2, mdot_out2=mdot_out2, beta2=beta2)

# Plot mdot_in vs redshift
plt.figure(figsize=(10,6))
plt.plot(redshifts1, mdot_in1, 'b-', label='NoBHFableHighSNEff')
plt.plot(redshifts2, mdot_in2, 'r-', label='NoBHFableHighSNEffHighRes')
plt.xlabel('Redshift')
plt.xlim(1,6)
plt.ylabel('$\dot(M)_{in}(R_{vir})$ [M$_\odot yr^{-1}$]')
plt.title('Gas Mass Inflow Rate vs Redshift')
plt.grid(True)
plt.legend()
plt.savefig('mass_inflow_rate.png')
plt.close()

# Plot mdot_out vs redshift
plt.figure(figsize=(10,6))
plt.plot(redshifts1, mdot_out1, 'b-', label='NoBHFableHighSNEff')
plt.plot(redshifts2, mdot_out2, 'r-', label='NoBHFableHighSNEffHighRes')
plt.xlabel('Redshift')
plt.xlim(1,6)
plt.ylabel('$\dot(M)_{out}(R_{vir})$ [M$_\odot yr^{-1}$]')
plt.title('Gas Mass Outflow Rate vs Redshift')
plt.grid(True)
plt.legend()
plt.savefig('mass_outflow_rate.png')
plt.close()

# Plot beta vs redshift
plt.figure(figsize=(10,6))
plt.plot(redshifts1, beta1, 'b-', label='NoBHFableHighSNEff')
plt.plot(redshifts2, beta2, 'r-', label='NoBHFableHighSNEffHighRes')
plt.xlabel('Redshift')
plt.xlim(1,6)
plt.ylabel(r'$\beta_\mathrm{out}(R_\mathrm{vir})$')
plt.title('Mass Loading Factor vs Redshift')
plt.grid(True)
plt.legend()
plt.savefig('mass_loading_factor.png')
plt.close()