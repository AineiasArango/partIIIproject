import matplotlib.pyplot as plt
import numpy as np
import read_fof_files as rff
import read_snap_files as rsf
import sys
from scipy import spatial
import cosmo_utils as cu
import mass_flow_rate_function as mfr
from astropy.cosmology import Planck15
import neighbour_variables_dg as nv

data = np.load('/data/ERCblackholes4/aasnha2/for_aineias/plots/smooth_and_inout_test_results.npz')
smooth_results = data['smooth_results']
inout_results = data['inout_results']

neighbour_poss = smooth_results[:,:,:3]
neighbour_v_gass = smooth_results[:,:,3:6]
neighbour_densitiess = smooth_results[:,:,6]

print(neighbour_poss.shape)
print(neighbour_v_gass.shape)
print(neighbour_densitiess.shape)

in_or_out = []
for neighbour_pos, neighbour_v_gas, neighbour_densities in zip(neighbour_poss, neighbour_v_gass, neighbour_densitiess):
    in_or_out.append(nv.in_or_out_2(neighbour_pos, neighbour_v_gas, neighbour_densities))

in_or_out = np.array(in_or_out)
print(in_or_out.shape)

print(sum(in_or_out==inout_results))