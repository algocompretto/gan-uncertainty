"""
Multiple-point statistics workflow with GAN images.
"""
#from mpslib.scikit-mps.mpslib import trainingimages
import matplotlib.pyplot as plt
import imageio
import glob

def plot(image, title: str):
    plt.figure(figsize=(16,9))
    plt.axis('off')
    plt.title(f'{title}')
    plt.imshow(image, cmap='gray')
    plt.show()

for image_path in glob.glob("TI/*.png"):
    image = imageio.imread(image_path)

plot(image, "Imagem original")