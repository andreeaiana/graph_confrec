import tensorflow as tf

from layers import Layer, Dense
from inits import glorot, zeros
import numpy as np

# DISCLAIMER:
# This code file is forked from https://github.com/oj9040/GraphSAGE_RL.


class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """
    def __init__(self, input_dim, output_dim, neigh_input_dim=None, dropout=0.,
                 bias=False, act=tf.nn.relu, name=None, concat=False,
                 **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs, att, num_nz = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - (1-self.dropout))
        self_vecs = tf.nn.dropout(self_vecs, 1 - (1-self.dropout))

        _, num_neigh, feature_dim = neigh_vecs.get_shape().as_list()
        neigh_means = tf.divide(
                tf.reduce_sum(input_tensor=neigh_vecs, axis=1),
                tf.tile(tf.expand_dims(num_nz+1e-10, -1), [1, feature_dim]))

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


# Low rank version of Mean Aggregator
class LRMeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None, dropout=0.,
                 bias=False, act=tf.nn.relu, name=None, concat=False,
                 **kwargs):
        super(LRMeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights1'] = glorot(
                    [neigh_input_dim, np.int(output_dim/3)],
                    name='neigh_weights1')
            self.vars['neigh_weights2'] = glorot(
                    [np.int(output_dim/3), output_dim], name='neigh_weights2')
            self.vars['self_weights1'] = glorot(
                    [input_dim, np.int(output_dim/3)], name='self_weights1')
            self.vars['self_weights2'] = glorot(
                    [np.int(output_dim/3), output_dim], name='self_weights2')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs, att, num_nz = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - (1-self.dropout))
        self_vecs = tf.nn.dropout(self_vecs, 1 - (1-self.dropout))

        _, num_neigh, feature_dim = neigh_vecs.get_shape().as_list()
        neigh_means = tf.divide(
                tf.reduce_sum(input_tensor=neigh_vecs, axis=1),
                tf.tile(tf.expand_dims(num_nz, -1), [1, feature_dim]))

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights1'])
        from_neighs = tf.matmul(from_neighs, self.vars['neigh_weights2'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights1"])
        from_self = tf.matmul(from_self, self.vars["self_weights2"])
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class LogicMeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None, dropout=0.,
                 bias=False, act=tf.nn.relu, name=None, concat=False,
                 **kwargs):
        super(LogicMeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs, att, num_nz = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - (1-self.dropout))
        self_vecs = tf.nn.dropout(self_vecs, 1 - (1-self.dropout))

        # modified
        _, num_neigh, feature_dim = neigh_vecs.get_shape().as_list()
        att = tf.reshape(att, [-1, num_neigh, 1])
        att = tf.tile(att, [1, 1, feature_dim])

        num_nz = tf.cast(num_nz, tf.float32) + 1e-10
        neigh_means = tf.divide(
                tf.reduce_sum(input_tensor=tf.multiply(
                        neigh_vecs, tf.cast(att, tf.float32)), axis=1),
                tf.tile(tf.expand_dims(num_nz, -1), [1, feature_dim]))

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class AttMeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None, dropout=0.,
                 bias=False, act=tf.nn.relu, name=None, concat=False,
                 **kwargs):
        super(AttMeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([neigh_input_dim, output_dim],
                                                name='neigh_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs, att, num_nz = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - (1-self.dropout))
        self_vecs = tf.nn.dropout(self_vecs, 1 - (1-self.dropout))

        # modified
        _, num_neigh, feature_dim = neigh_vecs.get_shape().as_list()
        att = tf.reshape(att, [-1, num_neigh, 1])
        att = tf.tile(att, [1, 1, feature_dim])
        neigh_means = tf.divide(
                tf.reduce_sum(input_tensor=tf.multiply(
                        neigh_vecs, att), axis=1),
                tf.tile(tf.expand_dims(num_nz, -1), [1, feature_dim]))

        # [nodes] x [out_dim]
        from_neighs = tf.matmul(neigh_means, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0., bias=False, act=tf.nn.relu, name=None,
                 concat=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, output_dim],
                                          name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):

        self_vecs, neigh_vecs, att, num_nz = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1 - (1-self.dropout))
        self_vecs = tf.nn.dropout(self_vecs, 1 - (1-self.dropout))

        _, num_neigh, feature_dim = neigh_vecs.get_shape().as_list()
        means = tf.divide(tf.reduce_sum(input_tensor=tf.concat(
                [neigh_vecs, tf.expand_dims(self_vecs, axis=1)], axis=1),
                axis=1),
                tf.tile(tf.expand_dims(1+num_nz+1e-10, -1), [1, feature_dim]))

        # [nodes] x [out_dim]
        output = tf.matmul(means, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, weight_decay, model_size="small",
                 neigh_input_dim=None, dropout=0., bias=False, act=tf.nn.relu,
                 name=None, concat=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(
                input_dim=neigh_input_dim,
                output_dim=hidden_dim,
                weight_decay=weight_decay,
                act=tf.nn.relu,
                dropout=dropout,
                sparse_inputs=False,
                logging=self.logging))

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs, att, num_nz = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(input=neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors,
                                          self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors,
                                          self.hidden_dim))
        neigh_h = tf.reduce_max(input_tensor=neigh_h, axis=1)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class MeanPoolingAggregator(Layer):
    """ Aggregates via mean-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, weight_decay, model_size="small",
                 neigh_input_dim=None, dropout=0., bias=False, act=tf.nn.relu,
                 name=None, concat=False, **kwargs):
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(
                input_dim=neigh_input_dim,
                output_dim=hidden_dim,
                weight_decay=weight_decay,
                act=tf.nn.relu,
                dropout=dropout,
                sparse_inputs=False,
                logging=self.logging))

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs, att, num_nz = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(input=neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors,
                                          self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors,
                                          self.hidden_dim))
        neigh_h = tf.reduce_mean(input_tensor=neigh_h, axis=1)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class TwoMaxLayerPoolingAggregator(Layer):
    """ Aggregates via pooling over two MLP functions.
    """
    def __init__(self, input_dim, output_dim, weight_decay, model_size="small",
                 neigh_input_dim=None, dropout=0., bias=False, act=tf.nn.relu,
                 name=None, concat=False, **kwargs):
        super(TwoMaxLayerPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim_1 = self.hidden_dim_1 = 512
            hidden_dim_2 = self.hidden_dim_2 = 256
        elif model_size == "big":
            hidden_dim_1 = self.hidden_dim_1 = 1024
            hidden_dim_2 = self.hidden_dim_2 = 512

        self.mlp_layers = []
        self.mlp_layers.append(Dense(
                input_dim=neigh_input_dim,
                output_dim=hidden_dim_1,
                weight_decay=weight_decay,
                act=tf.nn.relu,
                dropout=dropout,
                sparse_inputs=False,
                logging=self.logging))
        self.mlp_layers.append(Dense(
                input_dim=hidden_dim_1,
                output_dim=hidden_dim_2,
                weight_decay=weight_decay,
                act=tf.nn.relu,
                dropout=dropout,
                sparse_inputs=False,
                logging=self.logging))

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim_2, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

        dims = tf.shape(input=neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors,
                                          self.neigh_input_dim))

        for l in self.mlp_layers:
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors,
                                          self.hidden_dim_2))
        neigh_h = tf.reduce_max(input_tensor=neigh_h, axis=1)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class SeqAggregator(Layer):
    """ Aggregates via a standard LSTM.
    """
    def __init__(self, input_dim, output_dim, model_size="small",
                 neigh_input_dim=None, dropout=0., bias=False, act=tf.nn.relu,
                 name=None,  concat=False, **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256

        with tf.compat.v1.variable_scope(self.name + name + '_vars'):
            self.vars['neigh_weights'] = glorot([hidden_dim, output_dim],
                                                name='neigh_weights')

            self.vars['self_weights'] = glorot([input_dim, output_dim],
                                               name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        dims = tf.shape(input=neigh_vecs)
        batch_size = dims[0]
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        used = tf.sign(tf.reduce_max(input_tensor=tf.abs(neigh_vecs), axis=2))
        length = tf.reduce_sum(input_tensor=used, axis=1)
        length = tf.maximum(length, tf.constant(1.))
        length = tf.cast(length, tf.int32)

        with tf.compat.v1.variable_scope(self.name) as scope:
            try:
                rnn_outputs, rnn_states = tf.compat.v1.nn.dynamic_rnn(
                        self.cell,
                        neigh_vecs,
                        initial_state=initial_state,
                        dtype=tf.float32,
                        time_major=False,
                        sequence_length=length)
            except ValueError:
                scope.reuse_variables()
                rnn_outputs, rnn_states = tf.compat.v1.nn.dynamic_rnn(
                        self.cell,
                        neigh_vecs,
                        initial_state=initial_state,
                        dtype=tf.float32,
                        time_major=False,
                        sequence_length=length)
        batch_size = tf.shape(input=rnn_outputs)[0]
        max_len = tf.shape(input=rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        neigh_h = tf.gather(flat, index)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])

        output = tf.add_n([from_self, from_neighs])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)