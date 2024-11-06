import matplotlib.pyplot as plt
import numpy as np
import read_fof_files as rff
import read_snap_files as rsf
import sys
from scipy import spatial
import cosmo_utils as cu
import mass_flow_rate_function as mfr
from astropy.cosmology import Planck15

dir1="/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEff"
dir2="/data/ERCblackholes4/sk939/for_aineias/NoBHFableHighSNEffHighRes"
snap = 86

def get_redshift(snap_dir, snap_number):
    fof_file = rff.get_fof_filename(snap_dir, snap_number)
    redshift = rff.get_attribute(fof_file, "Redshift") #redshift
    return redshift

def printred():
    print(get_redshift(dir1, snap), get_redshift(dir2, snap), Planck15.age(get_redshift(dir1, snap)).value, Planck15.age(get_redshift(dir2, snap)).value)
def print_time(snap):
    print(Planck15.age(get_redshift(dir1, snap)).value, Planck15.age(get_redshift(dir2, snap)).value)

