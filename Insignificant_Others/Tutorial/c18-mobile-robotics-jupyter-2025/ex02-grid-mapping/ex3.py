#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh

def plot_gridmap(gridmap):
    """
    Display the gridmap as a grayscale image (0=black, 1=white).
    """
    plt.figure()
    plt.imshow(gridmap, cmap='Greys', vmin=0, vmax=1)
    plt.title("Occupancy Grid")
    plt.colorbar()

def init_gridmap(size, res):
    """
    Initialize a square gridmap of 'size' x 'size' meters,
    each cell of width 'res'.
    Returns a 2D numpy array, initialized to zeros.
    """
    gridmap = np.zeros([int(np.ceil(size/res)), int(np.ceil(size/res))])
    return gridmap

def world2map(pose, gridmap, map_res):
    """
    Convert continuous world coordinates (pose) to integer map indices.
    'pose' can be [x, y] or [x, y, theta].
    'gridmap' is used only to compute the origin as half of the shape.
    """
    origin = np.array(gridmap.shape) / 2
    new_pose = np.zeros_like(pose)
    # Round x, y to nearest cell index
    new_pose[0] = np.round(pose[0] / map_res) + origin[0]
    new_pose[1] = np.round(pose[1] / map_res) + origin[1]
    return new_pose.astype(int)

def v2t(pose):
    """
    Convert [x, y, theta] into a 3x3 homogeneous transform matrix.
    """
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([
        [c, -s,  pose[0]],
        [s,  c,  pose[1]],
        [0,  0,      1  ]
    ])
    return tr    

def ranges2points(ranges):
    """
    Convert a 1D array of range measurements (e.g., from a 2D LIDAR)
    into an array of 2D points in the sensor frame. Returns a 3xN
    array in homogeneous coordinates.
    """
    # Example laser properties
    start_angle = -1.5708     # ~ -90 degrees in radians
    angular_res = 0.0087270   # ~ 0.5 degrees
    max_range = 30.0          # sensor max range

    # Indices of valid beams (between 0 and max_range)
    num_beams = ranges.shape[0]
    valid_idx = (ranges > 0) & (ranges < max_range)

    # Angles for all beams; slice only valid beams
    angles = np.linspace(start_angle,
                         start_angle + num_beams * angular_res,
                         num_beams)[valid_idx]

    # X, Y in the sensor frame
    x_sens = ranges[valid_idx] * np.cos(angles)
    y_sens = ranges[valid_idx] * np.sin(angles)

    points = np.vstack((x_sens, y_sens))             # shape => 2xN
    points_hom = np.vstack((points, np.ones(points.shape[1])))  # shape => 3xN
    return points_hom

def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    """
    Convert valid range endpoints to map indices (2xN).
    - r_ranges: 1D array of beam ranges
    - w_pose:   [x, y, theta] in world coords
    """
    # Convert laser ranges to sensor-frame points
    r_points = ranges2points(r_ranges)

    # Transform from sensor to world
    w_T = v2t(w_pose)
    w_points = w_T @ r_points  # 3xN

    # Convert from world to map
    map_points = world2map(w_points, gridmap, map_res)  # shape => 3xN
    return map_points[0:2, :]  # only x,y indices

def poses2cells(w_pose, gridmap, map_res):
    """
    Convert a robot pose in world coords to map cells [row, col].
    """
    return world2map(w_pose, gridmap, map_res)

def bresenham(x0, y0, x1, y1):
    """
    Get the list of integer grid cells between (x0,y0) and (x1,y1)
    using Bresenham's line algorithm.
    """
    cells = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return cells

def prob2logodds(p):
    """
    Convert a probability p into log-odds:
       l = log( p / (1 - p) ).
    Clipped to avoid numerical issues if p=0 or p=1.
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def logodds2prob(l):
    """
    Convert log-odds l back to probability:
       p = 1 / (1 + exp(-l)).
    """
    return 1 / (1 + np.exp(-l))

def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
    """
    Inverse sensor model for a single cell:
      - If 'cell' == 'endpoint' => prob_occ
      - Else => prob_free
    """
    if np.array_equal(cell, endpoint):
        return prob_occ
    else:
        return prob_free

def grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap, map_res,
                                  prob_occ, prob_free, prior):
    """
    Occupancy grid mapping with known poses:
      - ranges_raw: [num_beams, num_scans] array of laser range data
      - poses_raw:  [num_scans, 3] array of [x, y, theta]
      - occ_gridmap: 2D array representing the occupancy grid (init with zeros)
      - map_res: resolution (meters per cell)
      - prob_occ, prob_free: sensor model probabilities
      - prior: prior occupancy probability

    Returns updated occupancy grid in probability space [0,1].
    """
    # 1. Initialize a log-odds map based on 'prior'
    log_odds_map = prob2logodds(prior) * np.ones_like(occ_gridmap)

    # 2. Iterate through each scan
    num_scans = poses_raw.shape[0]  # e.g. 361
    for t in range(num_scans):
        # a) Current robot pose => [x, y, theta]
        current_pose = poses_raw[t, :]

        # b) Ranges for this scan => ranges_raw[:, t], so shape is [num_beams]
        # Make sure ranges_raw.shape[1] == num_scans
        r_ranges = ranges_raw[:, t]

        # c) Convert endpoints to map cells
        endpoints = ranges2cells(r_ranges, current_pose, occ_gridmap, map_res)
        # d) Convert current pose to map cell
        robot_cell = poses2cells(current_pose, occ_gridmap, map_res)

        # e) For each endpoint, trace the free cells via Bresenham
        for i in range(endpoints.shape[1]):
            end_cell = endpoints[:, i]  # [x, y]
            ray_cells = bresenham(robot_cell[0], robot_cell[1],
                                  end_cell[0], end_cell[1])

            # f) Update log-odds for each cell in the ray
            for cell_ij in ray_cells:
                x_c, y_c = cell_ij
                # Skip if out of map bounds
                if (x_c < 0 or x_c >= occ_gridmap.shape[1] or
                    y_c < 0 or y_c >= occ_gridmap.shape[0]):
                    continue

                # Inverse sensor model
                p_inv = inv_sensor_model(cell_ij, end_cell, prob_occ, prob_free)

                # Update log-odds
                log_odds_map[y_c, x_c] += prob2logodds(p_inv)

    # 3. Convert final log-odds map to probabilities
    final_map = logodds2prob(log_odds_map)
    return final_map
