import numpy as np
import matplotlib.pyplot as plt
import random
import os

from IPython.display import display, clear_output
from PIL import Image, ImageSequence

#animate loss landscape over different signal fractions
def create_gif_nofit(m1, m2, z):
    
    output_directory = '2dhist_images'
    os.makedirs(output_directory, exist_ok=True)
    
    sig_space = np.logspace(-3, -1 , 20)
    
    frames = []
    for sb in sig_space:
    
        loss_landscape_nofit(sb, m1, m2, z)

        image_path = os.path.join(output_directory, 'hist_{}.png'.format(sb))
        plt.savefig(image_path)
        plt.close()
        clear_output(wait=True)

        # Append the image to the frames list
        frames.append(Image.open(image_path))

    # Create the final GIF that combines all frames
    output_gif_filename = 'sigspace{}{}fixed.gif'.format(m1, m2)
    frames[0].save(output_gif_filename, save_all=True, append_images=frames[1:], duration=400, loop=0)