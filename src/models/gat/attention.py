# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

# DISCLAIMER:
# This code file is derived from https://github.com/PetarV-/GAT,
# which is under an identical MIT license as graph_confrec.
# Conversion of original code to TF 2.0 is inspired by
# https://github.com/calciver/Graph-Attention-Networks/blob/master/Tensorflow_2_0_Graph_Attention_Networks_(GAT).ipynb


class attn_head(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, nb_nodes=None, in_drop=0.0, coef_drop=0.0,
                 activation=tf.nn.elu, residual=False):
        super(attn_head, self).__init__()
        self.activation = activation
        self.residual = residual

        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)

        self.conv_no_bias = tf.keras.layers.Conv1D(
                hidden_dim, 1, use_bias=False)
        self.conv_f1 = tf.keras.layers.Conv1D(1, 1)
        self.conv_f2 = tf.keras.layers.Conv1D(1, 1)

        self.conv_residual = tf.keras.layers.Conv1D(hidden_dim, 1)
        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))

    def __call__(self, seq, bias_mat, training):
        seq = self.in_dropout(seq, training=training)
        seq_fts = self.conv_no_bias(seq)
        f_1 = self.conv_f1(seq_fts)
        f_2 = self.conv_f2(seq_fts)

        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        coefs = self.coef_dropout(coefs, training=training)
        seq_fts = self.in_dropout(seq_fts, training=training)

        vals = tf.matmul(coefs, seq_fts)
        vals = tf.cast(vals, dtype=tf.float32)
        ret = vals + self.bias_zero

        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq)
            else:
                ret = ret + seq
        return self.activation(ret)


class sp_attn_head(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, nb_nodes, in_drop=0.0, coef_drop=0.0,
                 activation=tf.nn.elu, residual=False):
        super(sp_attn_head, self).__init__()
        self.hidden_dim = hidden_dim
        self.nb_nodes = nb_nodes
        self.activation = activation
        self.residual = residual

        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)

        self.conv_no_bias = tf.keras.layers.Conv1D(hidden_dim, 1,
                                                   use_bias=False)
        self.conv_f1 = tf.keras.layers.Conv1D(1, 1)
        self.conv_f2 = tf.keras.layers.Conv1D(1, 1)

        self.conv_residual = tf.keras.layers.Conv1D(hidden_dim, 1)
        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))

    def __call__(self, seq, bias_mat, training):
        adj_mat = bias_mat

        seq = self.in_dropout(seq, training=training)
        seq_fts = self.conv_no_bias(seq)
        f_1 = self.conv_f1(seq_fts)
        f_2 = self.conv_f2(seq_fts)

        f_1 = tf.reshape(f_1, (self.nb_nodes, 1))
        f_2 = tf.reshape(f_2, (self.nb_nodes, 1))
        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1, 0])

        logits = tf.compat.v1.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(
                indices=logits.indices,
                values=tf.nn.leaky_relu(logits.values),
                dense_shape=logits.dense_shape)
        coefs = tf.compat.v2.sparse.softmax(lrelu)

        if training is not False:
            coefs = tf.SparseTensor(
                    indices=coefs.indices,
                    values=self.coef_dropout(coefs.values, training=training),
                    dense_shape=coefs.dense_shape)
            seq_fts = self.in_dropout(seq_fts, training=training)

        coefs = tf.compat.v2.sparse.reshape(coefs,
                                            [self.nb_nodes, self.nb_nodes])

        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse.sparse_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, self.nb_nodes, self.hidden_dim])

        ret = vals + self.bias_zero

        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.conv_residual(seq)
            else:
                ret = ret + seq
        return self.activation(ret)
