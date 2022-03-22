"""
Multiple-point statistics workflow with GAN images.
"""
from g2s import g2s
import matplotlib.pyplot as plt
from time import time
import numpy
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

def load_sample(sample_path: str) -> numpy.array():
    return 

# create empty grid and add conditioning data
conditioning = numpy.zeros((150, 150))*numpy.nan; 
# fill the grid with 50 random points
conditioning.flat[numpy.random.permutation(conditioning.size)[:50]]=image.flat[numpy.random.permutation(image.size)[:50]];

# QS call using G2S
simulation,_=g2s('-a','qs', 
                 '-ti',image,
                 '-di',conditioning,
                 '-dt',[0],
                 '-k',1.2,
                 '-n',30,
                 '-j',0.5);