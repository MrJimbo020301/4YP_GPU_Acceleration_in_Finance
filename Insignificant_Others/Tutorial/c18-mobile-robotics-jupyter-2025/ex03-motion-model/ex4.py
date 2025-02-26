#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi 


def inverse_motion_model(prev_pose, cur_pose):
    
    x_prev, y_prev, theta_prev = prev_pose
    x_cur, y_cur, theta_cur = cur_pose
    
    delta_x = x_cur - x_prev 
    delta_y = y_cur - y_prev 

    rot1 = np.arctan2(delta_y, delta_x) - theta_prev 
    trans = np.sqrt(delta_x**2 + delta_y**2)
    rot2 = theta_cur - theta_prev - rot1 

    rot1 = normalize_angle(rot1)
    rot2 = normalize_angle(rot2)

    return rot1, trans, rot2


def prob(error, sigma):
    """
    Compute the probability density of a zero-mean Gaussian evaluated at error,
    with variance sigma.
    """
    return (1.0/np.sqrt(2*np.pi*sigma)) * np.exp(- (error**2)/(2*sigma))


def motion_model(cur_pose, prev_pose, odom, alpha):
    # Motion parameters from odometry (predicted)
    rot1_odom, trans_odom, rot2_odom = inverse_motion_model(prev_pose, odom)
    # Motion parameters from the actual pose transition 
    rot1_actual, trans_actual, rot2_actual = inverse_motion_model(prev_pose, cur_pose)

    # Compute the differences (errors) between the predicted and actual motion parameters
    error_rot1 = normalize_angle(rot1_actual - rot1_odom) 
    error_trans = trans_actual - trans_odom 
    error_rot2 = normalize_angle(rot2_actual - rot2_odom)

    # Compute variances based on the noise parameters and odometry values 
    sigma_rot1 = alpha[0] * rot1_odom**2 + alpha[1] * trans_odom**2 
    sigma_trans = alpha[2] * trans_odom**2 + alpha[3] * (rot1_odom + rot2_odom)**2
    sigma_rot2 = alpha[0] * rot2_odom**2 + alpha[1] * trans_odom**2

    p_rot1 = prob(error_rot1, sigma_rot1)
    p_trans = prob(error_trans, sigma_trans)
    p_rot2 = prob(error_rot2, sigma_rot2)

    # Total probability is the product of the individual probabilities 
    p = p_rot1 * p_trans * p_rot2 
    return p 


def sample(b):
    tot = 0
    for i in range(12):
        tot += np.random.uniform(0,1)
    tot = tot - 6 # Centre at zero (mean of 12 uniforms is 6)
    return tot * np.sqrt(b)


def sample_motion_model(prev_pose, odom, alpha):
    
    x,y,theta = prev_pose
    rot1, trans, rot2 = inverse_motion_model (prev_pose, odom)

    # Sample noise for each motin component 
    rot1_noise = sample(alpha[0] *rot1**2 + alpha[1]*trans**2)
    trans_noise = sample(alpha[2]*trans**2 + alpha[3]*(rot1**2 + rot2**2))
    rot2_noise = sample(alpha[0]*rot2**2 + alpha[1]*trans**2)

    rot1_hat = rot1 - rot1_noise 
    trans_hat = trans - trans_noise 
    rot2_hat  = rot2 - rot2_noise 

    # Compute new pose based on the noisy motion 
    x_new = x + trans_hat * np.cos(theta + rot1_hat)
    y_new = y + trans_hat * np.sin(theta + rot1_hat) 
    theta_new = normalize_angle(theta + rot1_hat + rot2_hat)

    return x_new, y_new, theta_new
