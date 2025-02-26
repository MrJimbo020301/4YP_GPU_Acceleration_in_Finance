# add your fancy code here
import numpy as np

def landmark_observation_model(z, x, b, sigma_r):
    # Compute the expected range as the Euclidean distance between the robot and the landmark
    expected_range = np.sqrt((b[0] - x[0]**2 + b[1] - x[1]**2))

    # Compute the Gaussion probability of the observed range given the expected range
    p = (1.0/np.sqrt(2 * np.pi * sigma_r**2)) * np.exp(-((z - expected_range)**2 /(2 * sigma_r**2)))                                    
    return p

def observation_likelihood(r, b, gridmap, sigma_r, size):
    rows, cols = gridmap.shape 

    sensor_x = (cols * size) / 2.0 
    sensor_y = (rows * size) / 2.0 

    # Set up ray-casting parameters 
    step = size / 2.0 
    current_range = 0.0 

    max_range = np.sqrt ((cols * size)**2 + (rows * size)**2)

    hit = False 
    while current_range < max_range:
        x_point = sensor_x + current_range * np.cos(b)
        y_point = sensor_y + current_range * np.sin(b) 

        i = int(y_point / size) 
        j = int(x_point / size) 

        if i < 0 or i >= rows or j < 0 or j >= cols: 
            break 

        # If an obstacle is detected (cell value == 1), break out 
        if gridmap[i, j] == 1: 
            hit = True
            break 

        current_range += step
    
    # If an obstacle was hit, current_range is our expected measurement; 
    # otherwise, use max_range 

    r_expected = current_range if hit else max_range 

    # compute the likelihood of the observed range given the expected range
    p = (1.0/np.sqrt(2 * np.pi * sigma_r**2)) * np.exp(-((r - r_expected)**2 /(2 * sigma_r**2)))
    return p