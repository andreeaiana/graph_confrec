from __future__ import division
from __future__ import print_function

from layers import Layer

import tensorflow as tf
import numpy as np
import pdb

# DISCLAIMER:
# This code file is forked from https://github.com/oj9040/GraphSAGE_RL


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

        # retrieve matrix of [numofids, degree(128)]
        adj_lists = tf.nn.embedding_lookup(params=self.adj_info, ids=ids)
        unif_rand = tf.random.uniform(
                [num_samples], minval=0, maxval=np.int(adj_lists.shape[1]),
                dtype=tf.int32)
        adj_lists = tf.gather(adj_lists, unif_rand, axis=1)

        condition = tf.equal(adj_lists, self.adj_info.shape[0]-1)
        case_true = tf.zeros(adj_lists.shape, tf.float32)
        case_false = tf.ones(adj_lists.shape, tf.float32)
        adj_lists_numnz = tf.reduce_sum(
                input_tensor=tf.compat.v1.where(
                        condition, case_true, case_false), axis=1)

        att_lists = tf.ones(adj_lists.shape)
        dummy_ = adj_lists_numnz

        return adj_lists, att_lists, adj_lists_numnz, dummy_


class MLNeighborSampler(Layer):
    """
    Sampling by regressor trained by RL-learning
    """
    def __init__(self, adj_info, features, max_degree, nonlinear_sampler,
                 **kwargs):
        super(MLNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32),
                                    trainable=False)
        self.batch_size = max_degree
        self.nonlinear_sampler = nonlinear_sampler
        self.node_dim = features.shape[1]
        self.reuse = False

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(params=self.adj_info, ids=ids)
        vert_num = ids.shape[0]
        neig_num = self.adj_info.shape[1]

        # build model
        # l = W*x1
        # l = relu(l*x2^t)
        with tf.compat.v1.variable_scope("MLsampler"):
            v_f = tf.nn.embedding_lookup(params=self.features, ids=ids)
            n_f = tf.nn.embedding_lookup(params=self.features,
                                         ids=tf.reshape(adj_lists, [-1]))
            # debug
            node_dim = np.int(v_f.shape[1])
            n_f = tf.reshape(n_f, shape=[-1, neig_num, node_dim])
            if self.nonlinear_sampler == True:
                v_f = tf.tile(tf.expand_dims(v_f, axis=1), [1, neig_num, 1])
                l = tf.compat.v1.layers.dense(
                        tf.concat([v_f, n_f], axis=2), 1,
                        activation=tf.nn.relu, trainable=False,
                        reuse=self.reuse, name='dense')
                out = tf.exp(l)
            else:
                l = tf.compat.v1.layers.dense(
                        v_f, self.node_dim, activation=None, trainable=False,
                        reuse=self.reuse, name='dense')
                l = tf.expand_dims(l, axis=1)
                l = tf.matmul(l, n_f, transpose_b=True, name='matmul')
                out = tf.nn.relu(l, name='relu')
            out = tf.squeeze(out)

            # sort (sort top k of negative estimated loss)
            out, idx_y = tf.nn.top_k(-out, num_samples)
            idx_y = tf.cast(idx_y, tf.int32)

            idx_x = tf.range(vert_num)
            idx_x = tf.tile(tf.expand_dims(idx_x, -1), [1, num_samples])

            adj_lists = tf.gather_nd(adj_lists, tf.stack(
                    [tf.reshape(idx_x, [-1]), tf.reshape(idx_y, [-1])],
                    axis=1))
            adj_lists = tf.reshape(adj_lists, [vert_num, num_samples])

            condition = tf.equal(adj_lists, self.adj_info.shape[0]-1)
            case_true = tf.zeros(adj_lists.shape, tf.float32)
            case_false = tf.ones(adj_lists.shape, tf.float32)
            adj_lists_numnz = tf.reduce_sum(
                    input_tensor=tf.compat.v1.where(
                            condition, case_true, case_false), axis=1)

            att_lists = tf.ones(adj_lists.shape)
            dummy_ = adj_lists_numnz

            self.reuse = True

        return adj_lists, att_lists, adj_lists_numnz, dummy_


class FastMLNeighborSampler(Layer):

    """
    Fast ver. of Sampling by regressor trained by RL-learning
    Replaced sorting operation with batched arg operation

    """

    def __init__(self, adj_info, features, max_degree, nonlinear_sampler,
                 **kwargs):
        super(FastMLNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32),
                                    trainable=False)
        self.batch_size = max_degree
        self.nonlinear_sampler = nonlinear_sampler
        self.node_dim = features.shape[1]
        self.reuse = False

    def _call(self, inputs):
        ids, num_samples = inputs

        adj_lists = tf.nn.embedding_lookup(params=self.adj_info, ids=ids)
        neig_num = np.int(self.adj_info.shape[1])

        adj_lists = tf.slice(adj_lists, [0, 0],
                             [-1, np.int(neig_num/num_samples)*num_samples])
        vert_num = np.int(adj_lists.shape[0])
        neig_num = np.int(adj_lists.shape[1])

        # build model
        # l = W*x1
        # l = relu(l*x2^t)
        with tf.compat.v1.variable_scope("MLsampler"):
            v_f = tf.nn.embedding_lookup(params=self.features, ids=ids)
            n_f = tf.nn.embedding_lookup(params=self.features,
                                         ids=tf.reshape(adj_lists, [-1]))

            # debug
            node_dim = np.int(v_f.shape[1])
            n_f = tf.reshape(n_f, shape=[-1, neig_num, node_dim])
            if self.nonlinear_sampler == True:
                v_f = tf.tile(tf.expand_dims(v_f, axis=1), [1, neig_num, 1])
                l = tf.compat.v1.layers.dense(
                        tf.concat([v_f, n_f], axis=2), 1,
                        activation=tf.nn.relu, trainable=False,
                        reuse=self.reuse, name='dense')
                out = tf.exp(l)
            else:
                l = tf.compat.v1.layers.dense(
                        v_f, node_dim, activation=None, trainable=False,
                        reuse=self.reuse, name='dense')
                l = tf.expand_dims(l, axis=1)
                l = tf.matmul(l, n_f, transpose_b=True, name='matmul')
                out = tf.nn.relu(l, name='relu')

            out = tf.squeeze(out)

            # group min
            group_dim = np.int(neig_num/num_samples)
            out = tf.reshape(out, [vert_num, num_samples, group_dim])
            idx_y = tf.argmin(input=out, axis=-1, output_type=tf.int32)
            delta = tf.expand_dims(
                    tf.range(0, group_dim*num_samples, group_dim), axis=0)
            delta = tf.tile(delta, [vert_num, 1])
            idx_y = idx_y + delta

            idx_x = tf.range(vert_num)
            idx_x = tf.tile(tf.expand_dims(idx_x, -1), [1, num_samples])

            adj_lists = tf.gather_nd(adj_lists,
                                     tf.stack([tf.reshape(idx_x, [-1]),
                                               tf.reshape(idx_y, [-1])],
                                              axis=1))
            adj_lists = tf.reshape(adj_lists, [vert_num, num_samples])

            condition = tf.equal(adj_lists, self.adj_info.shape[0]-1)
            case_true = tf.zeros(adj_lists.shape, tf.float32)
            case_false = tf.ones(adj_lists.shape, tf.float32)
            adj_lists_numnz = tf.reduce_sum(
                    input_tensor=tf.compat.v1.where(
                            condition, case_true, case_false), axis=1)

            att_lists = tf.ones(adj_lists.shape)
            dummy_ = adj_lists_numnz

            self.reuse = True

        return adj_lists, att_lists, adj_lists_numnz, dummy_
