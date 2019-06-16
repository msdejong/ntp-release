"""Score calculation for unification"""

import numpy as np
import tensorflow as tf

def representation_match(a, b, slope=1.0):

    """
    Calculates negative exponent of L2 distance between tensors
    Args:
    a: [N x k] tensor
    b: [M x k] tensor
    slope: coefficient on L2 distance in exponent
    Returns:
    sim: [N x M] tensor of negative exponent of L2 distances
    """

    c = -2 * tf.matmul(a, tf.transpose(b))
    na = tf.reduce_sum(tf.square(a), 1, keep_dims=True)
    nb = tf.reduce_sum(tf.square(b), 1, keep_dims=True)

    l2 = (c + tf.transpose(nb)) + na
    l2 = tf.clip_by_value(l2, 1e-6, 1000)
    l2 = tf.sqrt(l2)
    if slope != 1.0:
        sim = tf.exp(-l2 * slope)
    else:
        sim = tf.exp(-l2)

    return sim

def l2_sim_np(a, b):
    """Calculates negative exponent of L2 distances using numpy"""
    c = -2 * np.matmul(a, np.transpose(b))
    na = np.sum(a*a, axis=1)
    nb = np.sum(b*b, axis=1)
    # this is broadcasting!
    l2 = (c + np.transpose(nb)) + na
    l2 = np.clip(l2, 1e-6, 1000)
    l2 = np.sqrt(l2)
    sim = np.exp(-l2)

    return sim
