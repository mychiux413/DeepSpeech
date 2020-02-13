from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np


def cal_norm_length(lengths, alpha, beta, dynamic=True):
    if dynamic:
        return tf.pow((beta + tf.cast(lengths, dtype=tf.float32) / (beta + 1)), alpha)

    if isinstance(lengths, list):
        lengths = np.array(lengths, dtype=float)
    elif isinstance(lengths, int):
        lengths = float(lengths)
    return np.power((beta + lengths) / (beta + 1), alpha)
