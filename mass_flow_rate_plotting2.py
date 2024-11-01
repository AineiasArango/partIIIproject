import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
import scipy

import os
os.chdir("/home/aasnha2/Project/Plots")

# Load the data
data = np.load('mass_flow_rate_data.npz')

# Extract the data
redshifts1 = data['redshifts1']
mdot_tot1 = data['mdot_tot1']
mdot_in1 = np.abs(data['mdot_in1'])
mdot_out1 = data['mdot_out1']
beta1 = data['beta1']
cosmic_time1 = Planck18.age(redshifts1).value

redshifts2 = data['redshifts2']
mdot_tot2 = data['mdot_tot2']
mdot_in2 = np.abs(data['mdot_in2'])
mdot_out2 = data['mdot_out2']
beta2 = data['beta2']
cosmic_time2 = Planck18.age(redshifts2).value

# Plot mdot_in vs cosmic time
plt.figure(figsize=(10,6))
plt.plot(cosmic_time1, mdot_in1, 'b-', label='NoBHFableHighSNEff')
plt.plot(cosmic_time2, mdot_in2, 'r-', label='NoBHFableHighSNEffHighRes')
plt.xlabel('t [Gyr]')
plt.xlim(min(cosmic_time1.min(), cosmic_time2.min()), max(cosmic_time1.max(), cosmic_time2.max()))
plt.ylabel('$\dot{M}_{in}(R_{vir})$ [M$_\odot yr^{-1}$]')
plt.yscale('log')
plt.title('Gas Mass Inflow Rate vs Cosmic Time')
plt.legend()
plt.gca().invert_xaxis()  # This inverts the x-axis
plt.savefig('mass_inflow_rate_vs_cosmic_time.png')
plt.close()

# Plot mdot_out vs cosmic time
plt.figure(figsize=(10,6))
plt.plot(cosmic_time1, mdot_out1, 'b-', label='NoBHFableHighSNEff')
plt.plot(cosmic_time2, mdot_out2, 'r-', label='NoBHFableHighSNEffHighRes')  
plt.xlabel('t [Gyr]')
plt.xlim(min(cosmic_time1.min(), cosmic_time2.min()), max(cosmic_time1.max(), cosmic_time2.max()))
plt.ylabel('$\dot{M}_{out}(R_{vir})$ [M$_\odot yr^{-1}$]')
plt.yscale('log')
plt.title('Gas Mass Outflow Rate vs Cosmic Time')
plt.legend()
plt.gca().invert_xaxis()  # This inverts the x-axis
plt.savefig('mass_outflow_rate_vs_cosmic_time.png')
plt.close()

# Plot beta vs cosmic time
plt.figure(figsize=(10,6))
plt.plot(cosmic_time1, beta1, 'b-', label='NoBHFableHighSNEff')
plt.plot(cosmic_time2, beta2, 'r-', label='NoBHFableHighSNEffHighRes')  
plt.xlabel('t [Gyr]')
plt.xlim(min(cosmic_time1.min(), cosmic_time2.min()), max(cosmic_time1.max(), cosmic_time2.max()))
plt.ylabel(r'$\beta_{out}(R_{vir})$')
plt.yscale('log')
plt.title('Beta vs Cosmic Time')
plt.legend()
plt.gca().invert_xaxis()  # This inverts the x-axis
plt.savefig('beta_vs_cosmic_time.png')
plt.close()