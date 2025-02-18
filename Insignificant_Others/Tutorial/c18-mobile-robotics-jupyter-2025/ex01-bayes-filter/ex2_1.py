#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_belief(belief):
    plt.figure()
    
    ax = plt.subplot(2,1,1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0],1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")
    
    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")


def motion_model(action, belief):
    """
    Shifts the belief array based on the action (e.g., +1 for right, -1 for left).
    Wraps around the edges in this example; you can adjust to match your assignment.
    """
    new_belief = np.zeros_like(belief)
    n = len(belief)
    
    for i in range(n):
        # Shift index by 'action' (circularly, for example).
        # If your assignment requires a different approach (e.g., no wrap-around),
        # modify this logic accordingly.
        prev_idx = (i - action) % n
        new_belief[i] = belief[prev_idx]
    
    return new_belief


def sensor_model(observation, belief, world):
    # Example code:
    p_correct = 0.7
    p_incorrect = 0.9
    
    for i in range(len(belief)):
        if world[i] == observation:
            belief[i] *= p_correct
        else:
            belief[i] *= p_incorrect
    
    belief /= np.sum(belief)  # normalize
    return belief



def recursive_bayes_filter(actions, observations, belief, world):
    """
    Recursively applies the Bayesian filter for each time step:
      1) motion update (prediction)
      2) sensor update (correction)
    Returns a list of belief distributions at each step.
    """
    all_beliefs = []
    
    for u, z in zip(actions, observations):
        # Motion update
        belief = motion_model(u, belief)
        
        # Sensor update
        belief = sensor_model(z, belief, world)
        
        # Store the updated belief
        all_beliefs.append(belief.copy())
    
    return all_beliefs
