"""
Multiple-point statistics workflow with GAN images.
"""
from g2s import g2s
import argparse
import matplotlib.pyplot as plt
import time
import os
import cv2

from helpers.funcs import *

parser = argparse.ArgumentParser()
parser.add_argument("--conditional_data", type=str, default="data/conditioning_data/samples50", help="conditional data used for simulating")
parser.add_argument("--generative_model", type=str, default="wgan", help="number of epochs of training")
parser.add_argument("--output_folder", type=str, default=f"data/temp/simulated", help="output folder for all of the simulated images")
opt = parser.parse_args()
print(opt)

os.makedirs(f"{opt.output_folder}/{opt.generative_model}", exist_ok=True)

def timer(func):
    """
    Times the function passed as argument

    Args:
        func (`function object`): Function which you want to time.
    """
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

@timer
def simulate(image, conditioning):
    # QuickSampling call using G2S
    simulation, _ = g2s('-a', 'qs', 
                     '-ti', image,
                     '-di', conditioning,
                     '-dt', [1],
                     '-k', 64*64,
                     '-n', 64*64,
                     '-j', 0.5)
    
    plt.imshow(simulation, cmap="gray")
    plt.axis('off')
    plt.grid('off')
    plt.savefig(f'{opt.output_folder}/{opt.generative_model}/{time.time()}.png', dpi=300, bbox_inches='tight',
     transparent="True", pad_inches=0)


# Create the grid with loaded conditioning data
print("[INFO] Loading conditional data")
conditioning_dictionary = read_conditional_samples(opt.conditional_data)
conditioning = conditioning_dictionary['D']
conditioning = convert_to_grid(conditioning)
print("[INFO] Loaded conditional data!")

target_image = load_target_ti('strebelle')
path = "data/temp/selected"

for im in os.listdir(path+'/'):
    # Loading training image
    image = cv2.imread(f"{path}/{im}")
    simulate(image, conditioning)