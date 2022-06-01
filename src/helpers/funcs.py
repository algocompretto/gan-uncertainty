
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from torchvision.utils import save_image
from torch.autograd import Variable
import torch
import time
import numpy as np
import cv2
import os

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


def load_target_ti(fname):
    im = cv2.imread(f"data/TI/{fname}.png", cv2.COLOR_BGR2GRAY)
    # Binarization
    ret, target_img = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Resize image
    target_img = resize(target_img, (250, 250))
    return target_img


def check_similarity(image, target_img):
    return True if mean_squared_error(image, target_img) < 0.4 else False
    #return True if ssim(image, target_img, data_range=image.max()-image.min()) < 0.30 else False


def to_binary(filename):
    image = cv2.imread(f"{filename}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Binarization
    _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    return th

def generate_images(generator_path, shape, latent_dim,output_folder):
    # Load generator from saved model
    generator = torch.load(generator_path)
    generator.eval()

    # Generating images for evaluation
    z = Variable(torch.Tensor(np.random.normal(0, 1, (shape, latent_dim))))
    gen_imgs = generator(z)

    for idx, im in enumerate(gen_imgs):
        filename = f"{output_folder}/{time.time()}.png"
        save_image(im.data, filename)
        binary_image = to_binary(filename)

        try:
            dataset = np.loadtxt(f"bin/gan_results.out")
            numpy_tensor = binary_image.squeeze().ravel()
            new_TI = np.column_stack((dataset, numpy_tensor/255))
            np.savetxt(fname = "bin/gan_results.out",
                        X = new_TI)

        except FileNotFoundError:
            numpy_tensor = binary_image.squeeze().ravel()
            np.savetxt(fname = f"bin/gan_results.out",
                        X=numpy_tensor/255)