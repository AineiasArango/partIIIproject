import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data1= np.load('/home/aasnha2/Project/Plots/Neighbour_variable_plots/neighbour_variables_dg_NoBHFableHighSNEff.npz')
masses1 = data1['masses']         
positions1 = data1['positions']
densities1 = data1['densities']
velocities1 = data1['velocities'] 
temps1 = data1['temps']
n_neighbours1 = data1['n_neighbours']
r_vir1 = data1['r_vir']
redshifts1 = data1['redshifts']
radial_positions1 = np.linalg.norm(positions1, axis=1)
radial_velocities1 = np.sum(velocities1 * (positions1 / np.linalg.norm(positions1, axis=1)[:, np.newaxis]), axis=1)
radial_momentum1 = masses1 * radial_velocities1 

data2= np.load('/home/aasnha2/Project/Plots/Neighbour_variable_plots/neighbour_variables_dg_NoBHFableHighSNEffHighRes.npz')
masses2 = data2['masses']         
positions2 = data2['positions']
densities2 = data2['densities']
velocities2 = data2['velocities'] 
temps2 = data2['temps']
n_neighbours2 = data2['n_neighbours']
r_vir2 = data2['r_vir']
redshifts2 = data2['redshifts']
radial_positions2 = np.linalg.norm(positions2, axis=1)
radial_velocities2 = np.sum(velocities2 * (positions2 / np.linalg.norm(positions2, axis=1)[:, np.newaxis]), axis=1)
radial_momentum2 = masses2 * radial_velocities2 

import os
os.chdir("/data/ERCblackholes4/aasnha2/for_aineias/plots/")
#snapshots I want to use are at indices 3, 5, 14, 29

#Split masses1 into snapshots using n_neighbours1
masses1_split = []
start_idx = 0
for n in n_neighbours1:
    masses1_split.append(masses1[start_idx:start_idx + n])
    start_idx += n


# Create figure and axis
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts1[i]:.1f}' for i in snapshots]

# Plot KDE for all masses1
kde_all = stats.gaussian_kde(masses1)
x_range_all = np.linspace(min(masses1), max(masses1), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    masses = masses1_split[snapshot]
    kde = stats.gaussian_kde(masses)
    x_range = np.linspace(min(masses), max(masses), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Mass (M$_\odot$)')
ax.set_ylabel('Density')
ax.set_title('Mass Distribution at Different Redshifts')
ax.legend()

plt.tight_layout()
plt.savefig('mass_distribution_redshifts_sim1.png')
plt.close()

#Split densities1 into snapshots using n_neighbours1
densities1_split = []
start_idx = 0
for n in n_neighbours1:
    densities1_split.append(densities1[start_idx:start_idx + n])
    start_idx += n

# Create figure and axis
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts1[i]:.1f}' for i in snapshots]

# Plot KDE for all densities1
kde_all = stats.gaussian_kde(densities1)
x_range_all = np.linspace(min(densities1), max(densities1), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    densities = densities1_split[snapshot]
    kde = stats.gaussian_kde(densities)
    x_range = np.linspace(min(densities), max(densities), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Density (M$_\odot$/kpc$^3$)')
ax.set_ylabel('Density')
ax.set_title('Density Distribution at Different Redshifts')
ax.legend()

plt.tight_layout()
plt.savefig('density_distribution_redshifts_sim1.png')
plt.close()

#Split masses2 into snapshots using n_neighbours2
masses2_split = []
start_idx = 0
for n in n_neighbours2:
    masses2_split.append(masses2[start_idx:start_idx + n])
    start_idx += n

# Create figure and axis
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts2[i]:.1f}' for i in snapshots]

# Plot KDE for all masses2
kde_all = stats.gaussian_kde(masses2)
x_range_all = np.linspace(min(masses2), max(masses2), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    masses = masses2_split[snapshot]
    kde = stats.gaussian_kde(masses)
    x_range = np.linspace(min(masses), max(masses), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Mass (M$_\odot$)')
ax.set_ylabel('Density')
ax.set_title('Mass Distribution at Different Redshifts')
ax.legend()

plt.tight_layout()
plt.savefig('mass_distribution_redshifts_sim2.png')
plt.close()

#Split densities2 into snapshots using n_neighbours2
densities2_split = []
start_idx = 0
for n in n_neighbours2:
    densities2_split.append(densities2[start_idx:start_idx + n])
    start_idx += n

# Create figure and axis
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts2[i]:.1f}' for i in snapshots]

# Plot KDE for all densities2
kde_all = stats.gaussian_kde(densities2)
x_range_all = np.linspace(min(densities2), max(densities2), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    densities = densities2_split[snapshot]
    kde = stats.gaussian_kde(densities)
    x_range = np.linspace(min(densities), max(densities), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Density (M$_\odot$/kpc$^3$)')
ax.set_ylabel('Density')
ax.set_title('Density Distribution at Different Redshifts')
ax.legend()

plt.tight_layout()
plt.savefig('density_distribution_redshifts_sim2.png')
plt.close()

# Create figure and axis for mass histogram with KDE
fig, ax = plt.subplots(figsize=(10,6))

# Plot histogram of masses1
ax.hist(masses1, bins=50, density=True, alpha=0.5, label='Histogram')

# Fit and plot KDE
kde = stats.gaussian_kde(masses1)
x_range = np.linspace(min(masses1), max(masses1), 200)
ax.plot(x_range, kde(x_range), 'r-', label='KDE', linewidth=2)

ax.set_xlabel('Mass (M$_\odot$)')
ax.set_ylabel('Density')
ax.set_title('Mass Distribution with KDE')
ax.legend()

plt.tight_layout()
plt.savefig('mass_distribution_kde_sim1.png')
plt.close()

#Split temps1 into snapshots using n_neighbours1
temps1_split = []
start_idx = 0
for n in n_neighbours1:
    temps1_split.append(temps1[start_idx:start_idx + n])
    start_idx += n

# Create figure and axis
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts1[i]:.1f}' for i in snapshots]

# Plot KDE for all temps1
kde_all = stats.gaussian_kde(temps1)
x_range_all = np.linspace(min(temps1), max(temps1), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    temps = temps1_split[snapshot]
    kde = stats.gaussian_kde(temps)
    x_range = np.linspace(min(temps), max(temps), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Density')
ax.set_title('Temperature Distribution at Different Redshifts')
ax.legend()

plt.tight_layout()
plt.savefig('temperature_distribution_redshifts_sim1.png')
plt.close()

#Split temps2 into snapshots using n_neighbours2
temps2_split = []
start_idx = 0
for n in n_neighbours2:
    temps2_split.append(temps2[start_idx:start_idx + n])
    start_idx += n

# Create figure and axis
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts2[i]:.1f}' for i in snapshots]

# Plot KDE for all temps2
kde_all = stats.gaussian_kde(temps2)
x_range_all = np.linspace(min(temps2), max(temps2), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    temps = temps2_split[snapshot]
    kde = stats.gaussian_kde(temps)
    x_range = np.linspace(min(temps), max(temps), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Density')
ax.set_title('Temperature Distribution at Different Redshifts')
ax.legend()

plt.tight_layout()
plt.savefig('temperature_distribution_redshifts_sim2.png')
plt.close()

#Split radial_positions1 into snapshots using n_neighbours1
radial_positions1_split = []
start_idx = 0
for n in n_neighbours1:
    radial_positions1_split.append(radial_positions1[start_idx:start_idx + n])
    start_idx += n

# Create figure and axis
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts1[i]:.1f}' for i in snapshots]

# Plot KDE for all radial_positions1
kde_all = stats.gaussian_kde(radial_positions1)
x_range_all = np.linspace(min(radial_positions1), max(radial_positions1), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    positions = radial_positions1_split[snapshot]
    kde = stats.gaussian_kde(positions)
    x_range = np.linspace(min(positions), max(positions), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Radial Position (kpc)')
ax.set_ylabel('Density')
ax.set_title('Radial Position Distribution at Different Redshifts')
ax.legend()

plt.tight_layout()
plt.savefig('radial_position_distribution_redshifts_sim1.png')
plt.close()

#Split radial_positions2 into snapshots using n_neighbours2
radial_positions2_split = []
start_idx = 0
for n in n_neighbours2:
    radial_positions2_split.append(radial_positions2[start_idx:start_idx + n])
    start_idx += n

# Create figure and axis
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts2[i]:.1f}' for i in snapshots]

# Plot KDE for all radial_positions2
kde_all = stats.gaussian_kde(radial_positions2)
x_range_all = np.linspace(min(radial_positions2), max(radial_positions2), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    positions = radial_positions2_split[snapshot]
    kde = stats.gaussian_kde(positions)
    x_range = np.linspace(min(positions), max(positions), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Radial Position (kpc)')
ax.set_ylabel('Density')
ax.set_title('Radial Position Distribution at Different Redshifts')
ax.legend()

plt.tight_layout()
plt.savefig('radial_position_distribution_redshifts_sim2.png')
plt.close()

#Split radial_velocities1 into snapshots using n_neighbours1
radial_velocities1_split = []
start_idx = 0
for n in n_neighbours1:
    radial_velocities1_split.append(radial_velocities1[start_idx:start_idx + n])
    start_idx += n

# Create figure and axis
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts1[i]:.1f}' for i in snapshots]

# Plot KDE for all radial_velocities1
kde_all = stats.gaussian_kde(radial_velocities1)
x_range_all = np.linspace(min(radial_velocities1), max(radial_velocities1), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    velocities = radial_velocities1_split[snapshot]
    kde = stats.gaussian_kde(velocities)
    x_range = np.linspace(min(velocities), max(velocities), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Radial Velocity (km/s)')
ax.set_ylabel('Density')
ax.set_title('Radial Velocity Distribution at Different Redshifts')
ax.legend()

plt.tight_layout()
plt.savefig('radial_velocity_distribution_redshifts_sim1.png')
plt.close()

#Split radial_velocities2 into snapshots using n_neighbours2
radial_velocities2_split = []
start_idx = 0
for n in n_neighbours2:
    radial_velocities2_split.append(radial_velocities2[start_idx:start_idx + n])
    start_idx += n

# Create figure and axis
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts2[i]:.1f}' for i in snapshots]

# Plot KDE for all radial_velocities2
kde_all = stats.gaussian_kde(radial_velocities2)
x_range_all = np.linspace(min(radial_velocities2), max(radial_velocities2), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    velocities = radial_velocities2_split[snapshot]
    kde = stats.gaussian_kde(velocities)
    x_range = np.linspace(min(velocities), max(velocities), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Radial Velocity (km/s)')
ax.set_ylabel('Density')
ax.set_title('Radial Velocity Distribution at Different Redshifts')
ax.legend()

plt.tight_layout()
plt.savefig('radial_velocity_distribution_redshifts_sim2.png')
plt.close()

#Split radial_momentum1 into snapshots using n_neighbours1
radial_momentum1_split = []
start_idx = 0
for n in n_neighbours1:
    radial_momentum1_split.append(radial_momentum1[start_idx:start_idx + n])
    start_idx += n

# Create figure and axis
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts1[i]:.1f}' for i in snapshots]

# Plot KDE for all radial_momentum1
kde_all = stats.gaussian_kde(radial_momentum1)
x_range_all = np.linspace(min(radial_momentum1), max(radial_momentum1), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    momentum = radial_momentum1_split[snapshot]
    kde = stats.gaussian_kde(momentum)
    x_range = np.linspace(min(momentum), max(momentum), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Radial Momentum (M$_\odot$ km/s)')
ax.set_ylabel('Density')
ax.set_title('Radial Momentum Distribution at Different Redshifts')
ax.legend()

plt.tight_layout()
plt.savefig('radial_momentum_distribution_redshifts_sim1.png')
plt.close()

#Split radial_momentum2 into snapshots using n_neighbours2
radial_momentum2_split = []
start_idx = 0
for n in n_neighbours2:
    radial_momentum2_split.append(radial_momentum2[start_idx:start_idx + n])
    start_idx += n

# Create figure and axis
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts2[i]:.1f}' for i in snapshots]

# Plot KDE for all radial_momentum2
kde_all = stats.gaussian_kde(radial_momentum2)
x_range_all = np.linspace(min(radial_momentum2), max(radial_momentum2), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    momentum = radial_momentum2_split[snapshot]
    kde = stats.gaussian_kde(momentum)
    x_range = np.linspace(min(momentum), max(momentum), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Radial Momentum (M$_\odot$ km/s)')
ax.set_ylabel('Density')
ax.set_title('Radial Momentum Distribution at Different Redshifts')
ax.legend()

plt.tight_layout()
plt.savefig('radial_momentum_distribution_redshifts_sim2.png')
plt.close()

# Create figure and axis for masses1
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts1[i]:.1f}' for i in snapshots]

# Plot KDE for all masses1
kde_all = stats.gaussian_kde(masses1)
x_range_all = np.linspace(min(masses1), max(masses1), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    masses = masses1_split[snapshot]
    kde = stats.gaussian_kde(masses)
    x_range = np.linspace(min(masses), max(masses), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Mass (M$_\odot$)')
ax.set_ylabel('Density')
ax.set_title('Mass Distribution at Different Redshifts')
ax.set_yscale('log')
ax.legend()

plt.tight_layout()
plt.savefig('mass_distribution_redshifts_sim1_log.png')
plt.close()

# Create figure and axis for masses2
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts2[i]:.1f}' for i in snapshots]

# Plot KDE for all masses2
kde_all = stats.gaussian_kde(masses2)
x_range_all = np.linspace(min(masses2), max(masses2), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    masses = masses2_split[snapshot]
    kde = stats.gaussian_kde(masses)
    x_range = np.linspace(min(masses), max(masses), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Mass (M$_\odot$)')
ax.set_ylabel('Density')
ax.set_title('Mass Distribution at Different Redshifts')
ax.set_yscale('log')
ax.legend()

plt.tight_layout()
plt.savefig('mass_distribution_redshifts_sim2_log.png')
plt.close()

# Create figure and axis for temperatures1
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts1[i]:.1f}' for i in snapshots]

# Plot KDE for all temps1
kde_all = stats.gaussian_kde(temps1)
x_range_all = np.linspace(min(temps1), max(temps1), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    temps = temps1_split[snapshot]
    kde = stats.gaussian_kde(temps)
    x_range = np.linspace(min(temps), max(temps), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Density')
ax.set_title('Temperature Distribution at Different Redshifts')
ax.set_yscale('log')
ax.legend()

plt.tight_layout()
plt.savefig('temperature_distribution_redshifts_sim1_log.png')
plt.close()

# Create figure and axis for temperatures2
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts2[i]:.1f}' for i in snapshots]

# Plot KDE for all temps2
kde_all = stats.gaussian_kde(temps2)
x_range_all = np.linspace(min(temps2), max(temps2), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    temps = temps2_split[snapshot]
    kde = stats.gaussian_kde(temps)
    x_range = np.linspace(min(temps), max(temps), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Density')
ax.set_title('Temperature Distribution at Different Redshifts')
ax.set_yscale('log')
ax.legend()

plt.tight_layout()
plt.savefig('temperature_distribution_redshifts_sim2_log.png')
plt.close()

# Create figure and axis for radial momentum1
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts1[i]:.1f}' for i in snapshots]

# Plot KDE for all radial_momentum1
kde_all = stats.gaussian_kde(radial_momentum1)
x_range_all = np.linspace(min(radial_momentum1), max(radial_momentum1), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    momentum = radial_momentum1_split[snapshot]
    kde = stats.gaussian_kde(momentum)
    x_range = np.linspace(min(momentum), max(momentum), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Radial Momentum (M$_\odot$ km/s)')
ax.set_ylabel('Density')
ax.set_title('Radial Momentum Distribution at Different Redshifts')
ax.set_yscale('log')
ax.legend()

plt.tight_layout()
plt.savefig('radial_momentum_distribution_redshifts_sim1_log.png')
plt.close()

# Create figure and axis for radial momentum2
fig, ax = plt.subplots(figsize=(10,6))

# Plot KDE for each snapshot
snapshots = [3, 5, 14, 29]
labels = [f'z = {redshifts2[i]:.1f}' for i in snapshots]

# Plot KDE for all radial_momentum2
kde_all = stats.gaussian_kde(radial_momentum2)
x_range_all = np.linspace(min(radial_momentum2), max(radial_momentum2), 200)
ax.plot(x_range_all, kde_all(x_range_all), label='Combined', linestyle='--', color='black')

# Plot KDE for individual snapshots
for i, snapshot in enumerate(snapshots):
    momentum = radial_momentum2_split[snapshot]
    kde = stats.gaussian_kde(momentum)
    x_range = np.linspace(min(momentum), max(momentum), 200)
    ax.plot(x_range, kde(x_range), label=labels[i])

ax.set_xlabel('Radial Momentum (M$_\odot$ km/s)')
ax.set_ylabel('Density')
ax.set_title('Radial Momentum Distribution at Different Redshifts')
ax.set_yscale('log')
ax.legend()

plt.tight_layout()
plt.savefig('radial_momentum_distribution_redshifts_sim2_log.png')
plt.close()








