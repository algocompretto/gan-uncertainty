"""
Multiple-point statistics workflow with GAN images.
"""
import mpslib.scikit.mpslib as mps
import matplotlib.pyplot as plt
from time import time
import numpy as np
import imageio
import glob

def timer(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def plot(image, title: str):
    plt.figure(figsize=(16,9))
    plt.axis('off')
    plt.title(f'{title}')
    plt.imshow(image, cmap='gray')
    plt.show()

def initialize_mps():
  O = mps.mpslib(method='mps_snesim_tree', verbose_level=-1, debug_level=-1)
  O.parameter_filename = 'mps.txt'
  O.par['n_real']= 100
  O.par['origin']=np.array([0,0,0])
  O.par['hard_data_fnam']='conditioning_data/samples50'
  O.par['simulation_grid_size']=np.array([150, 150, 1])
  return O

@timer
def mps_algo(TI, O):
    # Setting TI to memory
    O.ti = TI

    # Parallel
    O.run()

for image_path in glob.glob("TI/*.png"):
    image = imageio.imread(image_path)

#plot(image, "Imagem original")

mps_runner = initialize_mps()
mps_algo(image, mps_runner)