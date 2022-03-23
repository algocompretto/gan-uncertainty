"""
Multiple-point statistics workflow with GAN images.
"""
from g2s import g2s
import matplotlib.pyplot as plt
from time import time
import numpy as np
import imageio
import glob
import os

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

image = imageio.imread('data_generated/1647486253.6988351.png')

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
    map_dict = {0:1, 1:2}
    dataX = array[:, 0]
    dataY = array[:, 1]
    # Mapping values to categorical variables
    dataClass = np.vectorize(map_dict.get)(array[:,3])
    x_, x_idx = np.unique(np.ravel(dataX), return_inverse=True)
    y_, y_idx = np.unique(np.ravel(dataY), return_inverse=True)
    newArray = np.zeros((len(x_), len(y_)), dtype=dataClass.dtype)
    newArray[x_idx, y_idx] = np.ravel(dataClass)
    return newArray


# create the grid with conditioning data
conditioning_dictionary = read_conditional_samples('conditioning_data/samples50')
conditioning = conditioning_dictionary['D']
conditioning = convert_to_grid(conditioning)

# QS call using G2S
simulation, _ = g2s('-a', 'qs', 
                 '-ti', image,
                 '-di', conditioning,
                 '-dt', [3],
                 '-k', 8,
                 '-n', 36,
                 '-j', 0.5)


# Display results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(7,4),subplot_kw={'aspect':'equal'})
fig.suptitle('qs conditional simulation',size='xx-large',y=0.9)
ax1.imshow(image)
ax1.set_title('training image');
ax1.axis('off');
ax2.imshow(conditioning)
ax2.set_title('conditioning');
ax2.axis('off');
ax3.imshow(simulation)
ax3.set_title('simulation');
ax3.axis('off');
plt.savefig('simulation.png', dpi=300)
plt.show()