"""Visualization of input image, learned reward image and value images by using Visdom"""
import matplotlib.pyplot as plt
import scipy.misc
from visdom import Visdom
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--datafile', 
                    type=str, 
                    default='./learned_rewards_values_8x8.npz', 
                    help='Path to data file')

args = parser.parse_args()

# Image size of grid world
grid_imsize = 256

# Load images to numpy array
imgs = np.load(args.datafile).items()[0][1]

grid_image = imgs[:2]
reward_image = imgs[2]
value_images = imgs[3:]

# Create grid image in RGB
gridsize = grid_image.shape[1] # Either H or W to obtain in CHW/HWC
if grid_image.shape[0] == 2: # CHW
        grid_image = grid_image.transpose([1, 2, 0]) # Convert to HWC
# Create RGB channels
c1 = scipy.misc.imresize(arr=grid_image[:, :, 0], # obstacles
                         size=[grid_imsize, grid_imsize, 1], 
                         interp='nearest') 
c2 = scipy.misc.imresize(arr=grid_image[:, :, 1], # goal
                         size=[grid_imsize, grid_imsize, 1], 
                         interp='nearest') 
c3 = scipy.misc.imresize(arr=np.zeros([gridsize, gridsize]), # nothing
                         size=[grid_imsize, grid_imsize, 1], 
                         interp='nearest') 
# Combine as a RGB image
grid_image = np.flipud(np.stack([c1, c2, c3], 2))

grid_image = grid_image.transpose([2, 0, 1]) # TEMP

# Create a visdom object
vis = Visdom()

# Image for grid image
vis.image(grid_image, opts=dict(title='Grid world', caption='Test'))
        
# Heatmap for reward image
vis.heatmap(reward_image.squeeze(), 
            opts=dict(colormap='Electric'))

# Heatmap for value image
for img in value_images:
    vis.heatmap(img.squeeze(), opts=dict(colormap='Electric'))
    
print('Finshed. Please open visdom page.')