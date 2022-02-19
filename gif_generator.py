import glob
from PIL import Image

# filepaths
fp_in = "GAN_results/*.png"
path = "GAN_results/"
fp_out = "movie.gif"

import os

#files = os.listdir(path)
#for idx, file in enumerate(files):
#    name = str(file).split("_")[0].zfill(5)
#    os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(name).zfill(5), '.png'])))


# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=120, loop=0)
