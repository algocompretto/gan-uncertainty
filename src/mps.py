"""
Multiple-point statistics workflow with GAN images.
"""
import argparse

import matplotlib.pyplot as plt
from g2s import g2s
import pygeostat as gs
from helpers.funcs import *

parser = argparse.ArgumentParser()
parser.add_argument("--conditional_data", type=str, default="data/conditioning_data/samples50",
                    help="conditional data used for simulating")
parser.add_argument("--generative_model", type=str, default="wgan", help="number of epochs of training")
parser.add_argument("--output_folder", type=str, default=f"data/temp/simulated",
                    help="output folder for all of the simulated images")
opt = parser.parse_args()

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
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


@timer
def simulate(img: object, cond: object):
    # QuickSampling call using G2S
    simulation, _ = g2s('-a', 'qs',
                        '-ti', img,
                        '-di', cond,
                        '-dt', [1],
                        '-k', 64 * 64,
                        '-n', 64 * 64,
                        '-j', 0.5)

    plt.imshow(simulation, cmap="gray")
    plt.axis('off')
    plt.grid('off')
    plt.savefig(f'{opt.output_folder}/{opt.generative_model}/{time.time()}.png', dpi=300, bbox_inches='tight',
                transparent="True", pad_inches=0)


@timer
def snesim_simulation():
    ti_selection_p = gs.Program("bin/snesim.exe", parfile='param.par')
    # TODO: adaptar para linux
    parstr = """
                    Parameters for SNESIM
                  ********************
                  
    START OF PARAMETERS:
    bin/ti_strebelle.out                      - file with original data
    1  2  0  3                      - columns for x, y, z, variable
    2                             - number of categories
    0   1                        - category codes
    0.73  0.27                - (target) global pdf
    0                             - use (target) vertical proportions (0=no, 1=yes)
    bin/localprop.dat                  - file with target vertical proportions
    0                            - servosystem parameter (0=no correction)
    0                             - debugging level: 0,1,2,3
    bin/snesim.dbg                    - debugging file
    bin/snesim.out                    - file for simulation output
    100                             - number of realizations to generate
    250    0.5    1.0              - nx,xmn,xsiz
    250    0.5    1.0              - ny,ymn,ysiz
    1     0.5    1.0              - nz,zmn,zsiz
    303268                         - random number seed
    10                            - max number of conditioning primary data
    10                            - min. replicates number
    0     0                     - condtion to LP (0=no, 1=yes), flag for iauto
    1.0     1.0                     - two weighting factors to combine P(A|B) and P(A|C)
    bin/localprop.dat                 - file for local proportions
    0                             - condition to rotation and affinity (0=no, 1=yes)
    bin/rot_aff.dat                  - file for rotation and affinity
    3                             - number of affinity categories
    1.0  1.0  1.0                 - affinity factors (X,Y,Z)     
    1.0  0.6  1.0                 - affinity factors             
    1.0  2.0  1.0                 - affinity factors             
    5                             - number of multiple grids
    bin/train.dat                     - file for training image
    250  250  1                  - training image dimensions: nxtr, nytr, nztr
    1                             - column for training variable
    1.0   1.0   1.0             - maximum search radii (hmax,hmin,vert)
    0.0   0.0   0.0               - angles for search ellipsoid
    """

    ti_selection_p.run(parstr=parstr)


print("Started snesim sim")
snesim_simulation()
print("Ended snesim sim")

# Create the grid with loaded conditioning data
print("[INFO] Loading conditional data")
conditioning_dictionary = read_conditional_samples(opt.conditional_data)
conditioning = conditioning_dictionary['D']
conditioning = convert_to_grid(conditioning)
print("[INFO] Loaded conditional data!")

target_image = load_target_ti('strebelle')
path = "data/temp/selected"

# for im in os.listdir(path + '/'):
#    pass
# Loading training image
# image = cv2.imread(f"{path}/{im}")
# simulate(image, conditioning)
