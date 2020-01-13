from layers import GraphConvolution, GraphConvolutionSparse
from layers import InnerProductDecoder
import tensorflow as tf

# DISCLAIMER:
# This code file is derived from https://github.com/Ruiqi-Hu/ARGA,
# which is under an identical MIT license as graph_confrec.


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' \
                + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' \
                + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()
        variables = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class ARGA(Model):
    def __init__(self, placeholders, num_features, features_nonzero, hidden1,
                 hidden2, **kwargs):
        super(ARGA, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.build()

    def _build(self):
        with tf.compat.v1.variable_scope('Encoder', reuse=None):
            self.hidden1_layer = GraphConvolutionSparse(
                    input_dim=self.input_dim,
                    output_dim=self.hidden1,
                    adj=self.adj,
                    features_nonzero=self.features_nonzero,
                    act=tf.nn.relu,
                    dropout=self.dropout,
                    logging=self.logging,
                    name='e_dense_1')(self.inputs)
            self.noise = gaussian_noise_layer(self.hidden1_layer, 0.1)
            self.embeddings = GraphConvolution(
                    input_dim=self.hidden1,
                    output_dim=self.hidden2,
                    adj=self.adj,
                    act=lambda x: x,
                    dropout=self.dropout,
                    logging=self.logging,
                    name='e_dense_2')(self.noise)
            self.z_mean = self.embeddings
            self.reconstructions = InnerProductDecoder(
                    input_dim=self.hidden2,
                    act=lambda x: x,
                    logging=self.logging)(self.embeddings)


class ARVGA(Model):
    def __init__(self, placeholders, num_features, num_nodes,
                 features_nonzero, hidden1, hidden2, **kwargs):
        super(ARVGA, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.build()

    def _build(self):
        with tf.compat.v1.variable_scope('Encoder'):
            self.hidden1_layer = GraphConvolutionSparse(
                    input_dim=self.input_dim,
                    output_dim=self.hidden1,
                    adj=self.adj,
                    features_nonzero=self.features_nonzero,
                    act=tf.nn.relu,
                    dropout=self.dropout,
                    logging=self.logging,
                    name='e_dense_1')(self.inputs)
            self.z_mean = GraphConvolution(
                    input_dim=self.hidden1,
                    output_dim=self.hidden2,
                    adj=self.adj,
                    act=lambda x: x,
                    dropout=self.dropout,
                    logging=self.logging,
                    name='e_dense_2')(self.hidden1_layer)
            self.z_log_std = GraphConvolution(
                    input_dim=self.hidden1,
                    output_dim=self.hidden2,
                    adj=self.adj,
                    act=lambda x: x,
                    dropout=self.dropout,
                    logging=self.logging,
                    name='e_dense_3')(self.hidden1_layer)
            self.z = self.z_mean + tf.random.normal(
                    [self.n_samples, self.hidden2]) * tf.exp(self.z_log_std)
            self.reconstructions = InnerProductDecoder(
                    input_dim=self.hidden2,
                    act=lambda x: x,
                    logging=self.logging)(self.z)
            self.embeddings = self.z


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.compat.v1.variable_scope(name, reuse=None):
        # np.random.seed(1)
        tf.compat.v1.set_random_seed(1)
        weights = tf.compat.v1.get_variable(
                "weights", shape=[n1, n2],
                initializer=tf.compat.v1.random_normal_initializer(
                        mean=0., stddev=0.01))
        bias = tf.compat.v1.get_variable(
                "bias", shape=[n2],
                initializer=tf.compat.v1.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


class Discriminator(Model):
    def __init__(self, hidden1, hidden2, hidden3, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.act = tf.nn.relu

    def construct(self, inputs, reuse=False):
        with tf.compat.v1.variable_scope('Discriminator'):
            if reuse:
                tf.compat.v1.get_variable_scope().reuse_variables()
            # np.random.seed(1)
            tf.compat.v1.set_random_seed(1)
            dc_den1 = tf.nn.relu(dense(
                    inputs, self.hidden2, self.hidden3, name='dc_den1'))
            dc_den2 = tf.nn.relu(dense(
                    dc_den1, self.hidden3, self.hidden1, name='dc_den2'))
            output = dense(dc_den2, self.hidden1, 1, name='dc_output')
            return output


def gaussian_noise_layer(input_layer, std):
    noise = tf.random.normal(shape=tf.shape(input=input_layer),
                             mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise
