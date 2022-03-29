"""
Multiple-point statistics workflow with GAN images.
"""
from g2s import g2s
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import cv2

os.makedirs("simulated", exist_ok=True)

def timer(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


def read_conditional_samples(filename='eas.dat', nanval=-997799):
    debug_level = 1
    if not (os.path.isfile(filename)):
        print("Filename:'%s', does not exist" % filename)
    
    file = open(filename, "r")
    if (debug_level > 0):
        print("eas: file ->%20s" % filename)

    eas = {}
    eas['title'] = (file.readline()).strip('\n')

    if (debug_level > 0):
        print("eas: title->%20s" % eas['title'])

    dim_arr = eas['title'].split()
    if len(dim_arr) == 3:
        eas['dim'] = {}
        eas['dim']['nx'] = int(dim_arr[0])
        eas['dim']['ny'] = int(dim_arr[1])
        eas['dim']['nz'] = int(dim_arr[2])

    eas['n_cols'] = int(file.readline())
    
    eas['header'] = []
    for i in range(0, eas['n_cols']):
        # print (i)
        h_val = (file.readline()).strip('\n')
        eas['header'].append(h_val)

        if (debug_level>1):
            print("eas: header(%2d)-> %s" % (i,eas['header'][i]))

    file.close()

    try:
        eas['D'] = np.genfromtxt(filename, skip_header=2+eas['n_cols'])
        if (debug_level>1):
            print("eas: Read data from %s" % filename)
    except:
        print("eas: COULD NOT READ DATA FROM %s" % filename)
        

    # add NaN values
    try:
        eas['D'][eas['D']==nanval]=np.nan        
    except:
        print("eas: FAILED TO HANDLE NaN VALUES (%d(" % nanval)
    
    # If dimensions are given in title, then convert to 2D/3D array
    if "dim" in eas:
        if (eas['dim']['nz']==1):
            eas['Dmat']=eas['D'].reshape((eas['dim']['ny'],eas['dim']['nx']))
        elif (eas['dim']['nx']==1):
            eas['Dmat']=np.transpose(eas['D'].reshape((eas['dim']['nz'],eas['dim']['ny'])))
        elif (eas['dim']['ny']==1):
            eas['Dmat']=eas['D'].reshape((eas['dim']['nz'],eas['dim']['nx']))
        else:
            eas['Dmat']=eas['D'].reshape((eas['dim']['nz'],eas['dim']['ny'],eas['dim']['nx']))
            

        eas['Dmat']=eas['D'].reshape((eas['dim']['nx'],eas['dim']['ny'],eas['dim']['nz']))
        eas['Dmat']=eas['D'].reshape((eas['dim']['nz'],eas['dim']['ny'],eas['dim']['nx']))
      
        eas['Dmat'] = np.transpose(eas['Dmat'], (2,1,0))
         
        if (debug_level>0):
            print("eas: converted data in matrixes (Dmat)")

    eas['filename']=filename

    return eas


def convert_to_grid(array):
    # Initializing variables
    map_dict = {0:1, 1:2}
    dataX = array[:, 0]
    dataY = array[:, 1]

    # Mapping values to categorical variables
    dataClass = np.vectorize(map_dict.get)(array[:,3])

    # Creating the simulation grid
    x_, x_idx = np.unique(np.ravel(dataX), return_inverse=True)
    y_, y_idx = np.unique(np.ravel(dataY), return_inverse=True)
    newArray = np.zeros((len(x_), len(y_)), dtype=dataClass.dtype)*np.nan;
    newArray[x_idx, y_idx] = np.ravel(dataClass)
    return newArray


@timer
def simulate(image, conditioning):
    # QuickSampling call using G2S
    simulation, _ = g2s('-a', 'qs', 
                     '-ti', image,
                     '-di', conditioning,
                     '-dt', [1],
                     '-k', 150,
                     '-n', 150,
                     '-j', 0.5,
                     '-fs')

    plt.imshow(simulation, cmap="gray")
    plt.axis('off')
    plt.grid('off')
    plt.savefig(f'simulated/{time.time()}.png', dpi=300, bbox_inches='tight',
     transparent="True", pad_inches=0)


# Create the grid with loaded conditioning data
conditioning_dictionary = read_conditional_samples('conditioning_data/samples50')
conditioning = conditioning_dictionary['D']
conditioning = convert_to_grid(conditioning)


path = "data_generated"
for im in os.listdir(path+'/'):
    # Loading training image
    image = cv2.imread(f"{path}/{im}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarization
    thresh = cv2.adaptiveThreshold(blurred, 0.5,
                            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 10, 1)

    simulate(thresh, conditioning)