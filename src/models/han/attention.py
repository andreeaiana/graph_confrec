# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

# DISCLAIMER:
# This code file is derived from https://github.com/Jhy1993/HAN.


class attn_head(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, nb_nodes=None, in_drop=0.0, coef_drop=0.0,
                 activation=tf.nn.elu, residual=False, return_coef=False):
        super(attn_head, self).__init__()
        self.activation = activation
        self.residual = residual
        self.return_coef = return_coef

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

        if self.return_coef:
            return self.activation(ret), coefs
        else:
            return self.activation(ret)


class attn_head_const_1(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, nb_nodes=None, in_drop=0.0, coef_drop=0.0,
                 activation=tf.nn.elu, residual=False):
        super(attn_head, self).__init__()
        self.activation = activation
        self.residual = residual
        self.return_coef = return_coef

        self.in_dropout = tf.keras.layers.Dropout(in_drop)
        self.coef_dropout = tf.keras.layers.Dropout(coef_drop)

        self.conv_no_bias = tf.keras.layers.Conv1D(
                hidden_dim, 1, use_bias=False)
        self.conv_f2 = tf.keras.layers.Conv1D(1, 1)

        self.conv_residual = tf.keras.layers.Conv1D(hidden_dim, 1)
        self.bias_zero = tf.Variable(tf.zeros(hidden_dim))

    def __call__(self, seq, bias_mat, training):
        adj_mat = 1.0 - bias_mat / -1e9
        seq = self.in_dropout(seq, training=training)
        seq_fts = self.conv_no_bias(seq)
        f_2 = self.conv_f2(seq_fts)

        logits = adj_mat
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


class SimpleAttLayer(tf.keras.layers.Layer):
    def __init__(self, attention_size, time_major=False, return_alphas=False):
        super(SimpleAttLayer, self).__init__()
        self.attention_size = attention_size
        self.time_major = time_major
        self.return_alphas = return_alphas

    def __call__(self, inputs):
        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and
            # the backward RNN outputs.
            inputs = tf.concat(inputs, 2)
        if self.time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.transpose(inputs, [1, 0, 2])
        # D value - hidden size of the RNN layer
        hidden_size = inputs.shape[2]

        # Trainable parameters
        w_omega = tf.Variable(tf.random.normal(
                [hidden_size, self.attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random.normal([self.attention_size],
                                               stddev=0.1))
        u_omega = tf.Variable(tf.random.normal([self.attention_size],
                                               stddev=0.1))

        # Applying fully connected layer with non-linear activation to each
        # of the B*T timestamps; the shape of `v` is (B,T,D)*(D,A)=(B,T,A),
        # where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is
        # reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name="vu")  # (B, T) shape
        alphas = tf.nn.softmax(vu, name="alphas")  # (B, T) shape

        # output of (Bi-)RNN is reduced with attention vector;
        # the result has (B, D) shape
        output = tf.reduce_sum(
                input_tensor=inputs * tf.expand_dims(alphas, -1), axis=1)

        if not self.return_alphas:
            return output
        else:
            return output, alphas
