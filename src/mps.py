"""
Multiple-point statistics traditional workflow with original TI images.
"""
import os
import time
import argparse
import pygeostat as gs

parser = argparse.ArgumentParser()
parser.add_argument("--original_image", type=str, help="original training image")
parser.add_argument("--output_folder", type=str, default=f"data/temp/simulated/",
                    help="output folder for all of the simulated images")
opt = parser.parse_args()

os.makedirs(f"{opt.output_folder}", exist_ok=True)


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
def snesim_simulation():
    ti_selection_p = gs.Program("bin/snesim.exe", parfile='bin/snesim_par.par')
    parstr = """
                    Parameters for SNESIM
                  ********************
                  
    START OF PARAMETERS:
    bin/samples50			- file with original data
    1  2  3  4                    - columns for x, y, z, variable
    2                             - number of categories
    0   1                        - category codes
    0.73  0.27                - (target) global pdf
    0                             - use (target) vertical proportions (0=no, 1=yes)
    vertprop.dat                  - file with target vertical proportions
    0                            - servosystem parameter (0=no correction)
    -1                             - debugging level: 0,1,2,3
    bin/snesim.dbg                    - debugging file
    bin/snesim.out                    - file for simulation output
    2                             - number of realizations to generate
    250    0.5    1.0              - nx,xmn,xsiz
    250    0.5    1.0              - ny,ymn,ysiz
    1     0.5    1.0              - nz,zmn,zsiz
    303258                         - random number seed
    50                            - max number of conditioning primary data
    10                            - min. replicates number
    0     0                     - condtion to LP (0=no, 1=yes), flag for iauto
    1.0     1.0                     - two weighting factors to combine P(A|B) and P(A|C)
    localprop.dat                 - file for local proportions
    0                             - condition to rotation and affinity (0=no, 1=yes)
    rotangle.dat                  - file for rotation and affinity
    3                             - number of affinity categories
    1.0  1.0  1.0                 - affinity factors (X,Y,Z)     
    1.0  0.6  1.0                 - affinity factors             
    1.0  2.0  1.0                 - affinity factors             
    6                             - number of multiple grids
    bin/ti_strebelle.out                     - file for training image
    250  250  1                  - training image dimensions: nxtr, nytr, nztr
    0                             - column for training variable
    10.0   10.0   5.0             - maximum search radii (hmax,hmin,vert)
    0.0   0.0   0.0               - angles for search ellipsoid
    """

    ti_selection_p.run(parstr=parstr)


print("[INFO] Started SNESIM simulation")
snesim_simulation()