import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import random

from perlin_noise import perlin_noise
from flood_fill import flood_fill

def generate_grid(size=100):

    area = 0 
    height, width = size, size

    while area < 0.6:
        perlin_noise_grid = perlin_noise(width, height, scale=6)

        grid = (perlin_noise_grid > 0.1) * 1
        grid = np.float32(grid)
        
        grid_copy = grid.copy()
        flood_fill(grid_copy, start_point=(0,0), new_value=0.5)
        
        mask = (grid_copy == 0)
        grid = grid + mask
        
        area = (np.count_nonzero(grid==0)/(size*size))
    return grid

def red_to_blue_gradient(size):
    """Generates a red to blue color gradient.

    Args:
        size: The number of steps in the gradient.

    Returns:
        A numpy array representing the gradient as RGB values.
    """
    red_start = np.array([1, 0, 0])  # RGB for red
    blue_end = np.array([0, 0, 1])  # RGB for blue

    gradient_colors = np.linspace(red_start, blue_end, size)
    return gradient_colors

def hsv_to_rgb(h):
    hsv_color = np.uint8([[[h*179,255,255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
    return bgr_color/255

def hsv_gradient(start_hue, end_hue, steps):
    """
    Generates an HSV color gradient.

    Args:
        start_hue: Starting hue value (0-1).
        end_hue: Ending hue value (0-1).
        steps: Number of steps in the gradient.

    Returns:
        A NumPy array of RGB colors representing the gradient.
    """
    hue_values = np.linspace(start_hue, end_hue, steps)
    hue_values = [hsv_to_rgb(h) for h in hue_values]
    gradient = np.array(hue_values)
    return gradient

def plot_trajectory(grid, trajectory):
    transversed_grid = np.stack([grid,grid,grid], axis=-1)
    print(transversed_grid.shape, grid.shape)

    gradients = hsv_gradient(0,1,len(trajectory))
    gradients = red_to_blue_gradient(len(trajectory))
    for i,(x,y) in enumerate(trajectory):
        transversed_grid[x,y,:] = gradients[i]
    plt.imshow(transversed_grid)

## Spawning the agent
# - Sample random cell
# - if empty
#   - Spawn point
# - else
#   - sample again
def spawn_point(grid):
    """
    Spawn the agent at a random point in any point
    """
    h,w = grid.shape

    start_cell = 1
    while start_cell != 0:
        x = random.randint(0,w-1)
        y = random.randint(0,h-1)
        
        start_cell = grid[x,y]

    return np.array((x,y))