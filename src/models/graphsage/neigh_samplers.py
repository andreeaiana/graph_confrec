from __future__ import division
from __future__ import print_function

from layers import Layer

import tensorflow as tf

# DISCLAIMER:
# This code file is forked from https://github.com/williamleif/GraphSAGE,
# which is under an identical MIT license as graph_confrec.


"""
Classes that are used to sample node neighborhoods
"""


class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(params=self.adj_info, ids=ids)
        adj_lists = tf.transpose(a=tf.random.shuffle(
                tf.transpose(a=adj_lists)))
        adj_lists = tf.slice(adj_lists, [0, 0], [-1, num_samples])
        return adj_lists
