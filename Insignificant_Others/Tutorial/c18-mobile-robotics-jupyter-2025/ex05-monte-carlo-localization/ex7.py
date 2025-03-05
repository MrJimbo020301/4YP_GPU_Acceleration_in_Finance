# ex7.py
# -*- coding: utf-8 -*-
import pickle
import matplotlib.pyplot as plt
import numpy as np


def world2map(pose, gridmap, map_res):
    max_y = np.size(gridmap, 0) - 1
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0] / map_res)
    new_pose[1] = max_y - np.round(pose[1] / map_res)
    return new_pose.astype(int)


def v2t(pose):
    """Convert pose [x, y, theta] into a 3×3 homogeneous transform."""
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([
        [c, -s, pose[0]],
        [s,  c, pose[1]],
        [0,   0,      1]
    ])
    return tr


def t2v(tr):
    """Convert a 3×3 homogeneous transform into pose [x, y, theta]."""
    x = tr[0, 2]
    y = tr[1, 2]
    th = np.arctan2(tr[1, 0], tr[0, 0])
    v = np.array([x, y, th])
    return v


def ranges2points(ranges, angles):
    """
    Convert ranges + angles into 2D points in the sensor frame.
    Returns homogeneous coordinates shape=(3, N).
    """
    max_range = 80
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points in sensor frame
    points = np.array([
        ranges[idx] * np.cos(angles[idx]),
        ranges[idx] * np.sin(angles[idx])
    ])
    # Add a row of 1s to get homogeneous coordinates
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom


def ranges2cells(r_ranges, r_angles, w_pose, gridmap, map_res):
    """
    Convert a set of LIDAR-like measurements (ranges, angles) at a world pose w_pose
    into map cells. The map cells will be used for lookups in a likelihood map.
    """
    # 1) Ranges to sensor-frame points
    r_points = ranges2points(r_ranges, r_angles)
    # 2) Transform into the world frame using w_pose
    w_P = v2t(w_pose)
    w_points = w_P @ r_points
    # 3) Convert world-frame points into map coordinates
    m_points = world2map(w_points, gridmap, map_res)
    # Return only [row, col] indices
    m_points = m_points[0:2, :]
    return m_points


def poses2cells(w_pose, gridmap, map_res):
    """
    Convert a single world-frame pose (x, y, theta) to map coordinates [row, col].
    """
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose


def init_uniform(num_particles, img_map, map_res):
    """
    Initialize particle poses uniformly across the entire map area.
    Particles format: [x, y, theta, weight].
    """
    particles = np.zeros((num_particles, 4))
    # x, y in world coordinates
    particles[:, 0] = np.random.rand(num_particles) * np.size(img_map, 1) * map_res
    particles[:, 1] = np.random.rand(num_particles) * np.size(img_map, 0) * map_res
    # theta in [-pi..pi]
    particles[:, 2] = np.random.rand(num_particles) * 2.0 * np.pi - np.pi
    # initial weights set to 1
    particles[:, 3] = 1.0
    return particles


def plot_particles(particles, img_map, map_res):
    """Simple helper to plot particles on top of the given map."""
    plt.matshow(img_map, cmap="gray")
    max_y = np.size(img_map, 0) - 1
    xs = particles[:, 0] / map_res
    ys = max_y - particles[:, 1] / map_res
    plt.plot(xs, ys, '.b')
    plt.xlim([0, img_map.shape[1]])
    plt.ylim([0, img_map.shape[0]])
    plt.show()


# -------------------------------------------------------------------------
#  Below are the stubs you need to fill in.
# -------------------------------------------------------------------------

def wrapToPi(theta):
    """
    Wrap angle to the range [-pi, pi].
    """
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def sample_normal_distribution(b):
    tot = 0.0
    for _ in range(12):
        tot += np.random.rand()  # Generate a random float in [0, 1)
    return b * (tot - 6.0)



def forward_motion_model(x_robo, del_rot_1, del_trans, del_rot_2):
    """
    Deterministic "forward" motion model:
    x' = x + del_trans * cos(theta + del_rot_1)
    y' = y + del_trans * sin(theta + del_rot_1)
    theta' = theta + del_rot_1 + del_rot_2
    Return the resulting pose as a 3×3 homogeneous transform.
    """
    x = x_robo[0] + del_trans * np.cos(x_robo[2] + del_rot_1)
    y = x_robo[1] + del_trans * np.sin(x_robo[2] + del_rot_1)
    theta = x_robo[2] + del_rot_1 + del_rot_2
    theta = wrapToPi(theta)
    return v2t([x, y, theta])


def sample_motion_model_odometry(x_robo_prev, u, noise_parameters):
    """
    Sample-based motion model for odometry:
      - x_robo_prev : previous pose [x, y, theta]
      - u : [del_rot1, del_trans, del_rot2], the measured odometry
      - noise_parameters : [alpha1, alpha2, alpha3, alpha4]
    Return new pose [x, y, theta] as a vector (not a 3×3 transform).
    """
    alpha1, alpha2, alpha3, alpha4 = noise_parameters
    del_rot1, del_trans, del_rot2 = u

    # Add noise to each component (Probabilistic Robotics, Thrun et al.)
    del_rot1_noise = sample_normal_distribution(
        np.sqrt(alpha1 * del_rot1**2 + alpha2 * del_trans**2)
    )
    del_trans_noise = sample_normal_distribution(
        np.sqrt(alpha3 * del_trans**2 + alpha4 * (del_rot1**2 + del_rot2**2))
    )
    del_rot2_noise = sample_normal_distribution(
        np.sqrt(alpha1 * del_rot2**2 + alpha2 * del_trans**2)
    )

    # "Hat" variables incorporate random noise
    del_rot1_hat = del_rot1 + del_rot1_noise
    del_trans_hat = del_trans + del_trans_noise
    del_rot2_hat = del_rot2 + del_rot2_noise

    # Forward motion model in homogeneous form, then back to [x, y, theta]
    T_new = forward_motion_model(x_robo_prev, del_rot1_hat, del_trans_hat, del_rot2_hat)
    return t2v(T_new)  # return [x, y, theta]


def compute_weights(x_pose, z_obs, gridmap, likelihood_map, map_res):
    """
    Compute the importance weight for a single pose x_pose given the
    sensor reading z_obs = (ranges, angles).
    A typical approach: convert ranges into map cells, look up each cell
    in likelihood_map, and accumulate or average the probability.
    """
    r_ranges, r_angles = z_obs
    # Convert the scanned points into map indices
    m_points = ranges2cells(r_ranges, r_angles, x_pose, gridmap, map_res)

    weight_accumulator = 0.0
    valid_hits = 0

    for i in range(m_points.shape[1]):
        mx = m_points[0, i]
        my = m_points[1, i]
        # Check bounds
        if (0 <= mx < gridmap.shape[1]) and (0 <= my < gridmap.shape[0]):
            # Accumulate probabilities from the likelihood map
            prob = likelihood_map[my, mx]
            weight_accumulator += prob
            valid_hits += 1

    if valid_hits > 0:
        # Average probability across all valid hits
        return weight_accumulator / valid_hits
    else:
        # If no valid hits, assign a very small weight
        return 1e-9


def resample(particles, weights, gridmap):
    """
    Low-variance (a.k.a. stochastic universal) resampling.
    particles shape: (N,4), weights shape: (N,).
    Return an array of the same shape with the resampled particles.
    """
    N = len(weights)
    new_particles = np.zeros_like(particles)
    r = np.random.rand() * (1.0 / N)
    c = weights[0]
    i = 0

    for m in range(N):
        U = r + m * (1.0 / N)
        while U > c:
            i += 1
            c += weights[i]
        # Copy that particle over
        new_particles[m, :] = particles[i, :]

    return new_particles


def mc_localization(odom, z, num_particles, particles, noise, gridmap, likelihood_map, map_res, img_map):
    """
    Main loop of Monte Carlo Localization:
      - odom: list or array of odometry readings, each an array [del_rot1, del_trans, del_rot2]
      - z:    list or array of sensor readings, each is (ranges, angles)
      - noise: [alpha1, alpha2, alpha3, alpha4] for motion model
      - Loop over each time step:
          1) For each particle, sample the motion model
          2) Compute weights from sensor data
          3) Normalize, resample
    Returns the final set of particles.
    """
    for t in range(len(odom)):
        # 1) Sample new poses from motion model
        for i in range(num_particles):
            old_pose = particles[i, 0:3]
            u_t = odom[t]
            new_pose = sample_motion_model_odometry(old_pose, u_t, noise)
            particles[i, 0] = new_pose[0]
            particles[i, 1] = new_pose[1]
            particles[i, 2] = wrapToPi(new_pose[2])

        # 2) Compute importance weights
        all_weights = np.zeros(num_particles)
        for i in range(num_particles):
            all_weights[i] = compute_weights(
                particles[i, 0:3],
                z[t],
                gridmap,
                likelihood_map,
                map_res
            )

        # Avoid all-zero case
        all_weights += 1e-12
        # Normalize
        all_weights /= np.sum(all_weights)

        # Store weights in the 4th column
        particles[:, 3] = all_weights

        # 3) Resample
        particles = resample(particles, all_weights, gridmap)

    return particles
