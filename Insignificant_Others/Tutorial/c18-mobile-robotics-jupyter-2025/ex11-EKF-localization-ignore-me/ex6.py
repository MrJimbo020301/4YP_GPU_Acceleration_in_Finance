# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse


def plot_state(mu, S, M):

    # initialize figure
    ax = plt.gca()
    ax.set_xlim([np.min(M[:, 0]) - 2, np.max(M[:, 0]) + 2])
    ax.set_xlim([np.min(M[:, 1]) - 2, np.max(M[:, 1]) + 2])
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

    # ellipse orientation
    a = np.sqrt(d[0])
    b = np.sqrt(d[1])

    # compute ellipse orientation
    if (v[0, 0] == 0):
        theta = np.pi / 2
    else:
        theta = np.arctan2(v[0, 1], v[0, 0])

    # create an ellipse
    ellipse = Ellipse((mu[0], mu[1]),
                      width=a * 2,
                      height=b * 2,
                      angle=np.deg2rad(theta),
                      edgecolor='blue',
                      alpha=0.3)

    ax = plt.gca()

    return ax.add_patch(ellipse)


def wrapToPi(theta):
    while theta < -np.pi:
        theta = theta + 2 * np.pi
    while theta > np.pi:
        theta = theta - 2 * np.pi
    return theta


def inverse_motion_model(pose, pose_prev):
    # TODO: you can use your implementation of the function from ex3
    pass

def ekf_predict(mu, S, u, R):
    # TODO
    return mu, S

def ekf_correct(mu, S, z, Q, M):
    # TODO
    return mu, S

def run_ekf_localization(dataset, R, Q, verbose=False):
    # TODO
    # Initialize state variable
    mu = dataset['gt'][0]
    S = np.zeros([3, 3])

    # Read map
    M = dataset['M']

    # initialize figure
    plt.figure(10)
    axes = plt.gca()
    axes.set_xlim([0, 25])
    axes.set_ylim([0, 25])
    plt.plot(M[:, 0], M[:, 1], '^r')
    plt.title('EKF Localization')

    num_steps = len(dataset['gt'])
    for i in range(num_steps):
        # TODO: iterate over prediction/correction and plot
        pass

    plt.show()         
    return mu, S
