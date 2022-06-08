import mpslib as mps
from helpers.funcs import *

# Creating MPS instance
O = mps.mpslib(method='mps_snesim_tree')

# Initializing parameters for MPS object
TI, _ = mps.trainingimages.strebelle()

O.ti = TI

O.par['n_cond'] = 50
O.par['n_real'] = 100
O.par['simulation_grid_size'][0]=150
O.par['simulation_grid_size'][1]=150
O.par['simulation_grid_size'][2]=1
O.par['hard_data_fnam']='data/conditioning_data/samples50'

O.run_parallel()

O.plot_reals()

O.plot_etype()
