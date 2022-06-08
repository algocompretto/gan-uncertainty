import os

import mpslib as mps
import numpy as np
from tqdm import tqdm
from helpers.funcs import to_binary

os.makedirs(name='data/temp/eas', exist_ok=True)

# Creating MPS instance
MPS = mps.mpslib(method='mps_snesim_tree')
MPS.par['n_cond']: int = 50
MPS.par['n_real']: int = 105
MPS.par['simulation_grid_size'][0]: int = 150
MPS.par['simulation_grid_size'][1]: int = 150
MPS.par['simulation_grid_size'][2]: int = 1
MPS.par['hard_data_fnam']: str = 'data/conditioning_data/samples50'


def create_eas_files(data_dir: str = 'data/temp'):
    # Convert each image to EAS format
    for img_filename in os.listdir(f'{data_dir}/wgan'):
        ti_array: np.array = to_binary(f'{data_dir}/wgan/{img_filename}')
        mps.eas.write(ti_array.ravel(),
                      f'{data_dir}/eas/{img_filename.split(".png")[0]}.dat')


print('[INFO] Creating EAS files of selected TIs...')
create_eas_files()

print('[INFO] Performing SNESIM simulation on TIs...')
ti: dict = mps.eas.read(f'data/temp/eas/1654695964.3630815.dat')
MPS.ti: np.array = ti['D'].reshape(150, 150, -1)
MPS.run_parallel()

#for ti_matrix in tqdm(os.listdir('data/temp/eas/')):
#    ti: dict = mps.eas.read(f'data/temp/eas/{ti_matrix}')
#    MPS.ti: np.array = ti['D'].reshape(150, 150, -1)
#    MPS.run()
