# ex6.py
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse

def plot_state(mu, S, M):
    # initialize figure
    ax = plt.gca()
    ax.set_xlim([np.min(M[:, 0]) - 2, np.max(M[:, 0]) + 2])
    ax.set_ylim([np.min(M[:, 1]) - 2, np.max(M[:, 1]) + 2])
    plt.plot(M[:, 0], M[:, 1], '^r')
    plt.title('EKF Localization')

    # visualize result
    plt.plot(mu[0], mu[1], '.b')
    plot_2dcov(mu, S)
    plt.draw()
    plt.pause(0.01)

def plot_2dcov(mu, cov):
    # covariance only in x,y
    d, v = np.linalg.eig(cov[:-1, :-1])
    # ellipse axes lengths (1 std dev)
    a = np.sqrt(d[0])
    b = np.sqrt(d[1])
    # compute ellipse orientation
    if v[0, 0] == 0:
        theta = np.pi / 2
    else:
        theta = np.arctan2(v[0, 1], v[0, 0])
    # create an ellipse (convert theta to degrees)
    ellipse = Ellipse((mu[0], mu[1]),
                      width=2*a,
                      height=2*b,
                      angle=np.degrees(theta),
                      edgecolor='blue',
                      alpha=0.3)
    ax = plt.gca()
    return ax.add_patch(ellipse)

def wrapToPi(theta):
    while theta < -np.pi:
        theta += 2 * np.pi
    while theta > np.pi:
        theta -= 2 * np.pi
    return theta

def inverse_motion_model(pose, pose_prev):
    """
    Compute the relative motion (odometry measurement) from pose_prev to pose.
    Both poses are assumed to be [x, y, theta].
    Returns: [delta_rot1, delta_trans, delta_rot2]
    """
    dx = pose[0] - pose_prev[0]
    dy = pose[1] - pose_prev[1]
    delta_trans = np.sqrt(dx**2 + dy**2)
    delta_rot1 = wrapToPi(np.arctan2(dy, dx) - pose_prev[2])
    delta_rot2 = wrapToPi(pose[2] - pose_prev[2] - delta_rot1)
    return np.array([delta_rot1, delta_trans, delta_rot2])

def ekf_predict(mu, S, u, R):
    """
    EKF prediction step.
    mu: current state [x, y, theta]
    S: current covariance (3x3)
    u: control input [delta_rot1, delta_trans, delta_rot2]
    R: motion noise covariance (3x3)
    Returns: predicted state and covariance.
    """
    delta_rot1, delta_trans, delta_rot2 = u
    theta = mu[2]
    # Predicted state using the motion model
    mu_bar = np.array([
        mu[0] + delta_trans * np.cos(theta + delta_rot1),
        mu[1] + delta_trans * np.sin(theta + delta_rot1),
        wrapToPi(theta + delta_rot1 + delta_rot2)
    ])
    # Jacobian with respect to state
    G = np.array([
        [1, 0, -delta_trans * np.sin(theta + delta_rot1)],
        [0, 1,  delta_trans * np.cos(theta + delta_rot1)],
        [0, 0, 1]
    ])
    # Jacobian with respect to control
    V = np.array([
        [-delta_trans * np.sin(theta + delta_rot1), np.cos(theta + delta_rot1), 0],
        [ delta_trans * np.cos(theta + delta_rot1), np.sin(theta + delta_rot1), 0],
        [1, 0, 1]
    ])
    S_bar = G @ S @ G.T + V @ R @ V.T
    return mu_bar, S_bar

def ekf_correct(mu, S, z, Q, M):
    """
    EKF correction step.
    mu: predicted state [x, y, theta]
    S: predicted covariance (3x3)
    z: list of observations for this time step.
       Each observation is a 3-element vector: [landmark_id, range, bearing]
    Q: measurement noise covariance (2x2)
    M: map of landmarks (each row is [x, y])
    Returns: updated state and covariance.
    """
    for obs in z:
        j = int(obs[0])
        z_range = obs[1]
        z_bearing = obs[2]
        # Landmark coordinates
        lx = M[j, 0]
        ly = M[j, 1]
        dx = lx - mu[0]
        dy = ly - mu[1]
        q = dx**2 + dy**2
        sqrt_q = np.sqrt(q)
        # Predicted measurement
        z_hat = np.array([
            sqrt_q,
            wrapToPi(np.arctan2(dy, dx) - mu[2])
        ])
        # Innovation (measurement residual)
        z_diff = np.array([
            z_range - z_hat[0],
            wrapToPi(z_bearing - z_hat[1])
        ])
        # Jacobian H of the measurement function with respect to the state
        H = np.array([
            [-dx/sqrt_q, -dy/sqrt_q, 0],
            [dy/q,      -dx/q,      -1]
        ])
        # Kalman gain
        SHT = S @ H.T
        K = SHT @ np.linalg.inv(H @ SHT + Q)
        # Update state and covariance
        mu = mu + K @ z_diff
        mu[2] = wrapToPi(mu[2])
        S = (np.eye(3) - K @ H) @ S
    return mu, S

def run_ekf_localization(dataset, R, Q, verbose=False):
    """
    Run the full EKF localization over a dataset.
    dataset: a dictionary containing:
       - 'gt': ground truth poses (Nx3), used here to compute control inputs
       - 'M': map (landmarks, each row [x, y])
       - 'z': list of observations for each time step; each element is a list of observations,
              and each observation is [landmark_id, range, bearing]
    R: motion noise covariance (3x3)
    Q: measurement noise covariance (2x2)
    Returns: final estimated state and covariance.
    """
    # Initialize state with the first ground truth pose
    gt = dataset['gt']
    mu = gt[0].copy()
    S = np.zeros([3, 3])
    M = dataset['M']

    # Initialize figure for plotting
    plt.figure(10)
    axes = plt.gca()
    axes.set_xlim([np.min(M[:, 0]) - 2, np.max(M[:, 0]) + 2])
    axes.set_ylim([np.min(M[:, 1]) - 2, np.max(M[:, 1]) + 2])
    plt.plot(M[:, 0], M[:, 1], '^r')
    plt.title('EKF Localization')

    num_steps = len(gt)
    for i in range(1, num_steps):
        # Prediction: compute control input from previous and current ground truth
        u = inverse_motion_model(gt[i], gt[i-1])
        mu, S = ekf_predict(mu, S, u, R)
        # Correction: if there are measurements for this time step, update with them
        if 'z' in dataset and len(dataset['z']) > i:
            z = dataset['z'][i]
            mu, S = ekf_correct(mu, S, z, Q, M)
        if verbose:
            print(f"Step {i}: mu = {mu}")
        # Plot current state estimate
        plot_state(mu, S, M)
    plt.show()
    return mu, S
