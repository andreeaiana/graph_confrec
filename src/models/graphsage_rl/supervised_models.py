import numpy as np
import tensorflow as tf

import models as models
import layers as layers
from aggregators import MeanAggregator, MaxPoolingAggregator
from aggregators import MeanPoolingAggregator, SeqAggregator, GCNAggregator

# DISCLAIMER:
# This code file is derived from https://github.com/oj9040/GraphSAGE_RL.


class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes, placeholders, features, adj, degrees,
                 layer_infos, weight_decay, learning_rate, batch_size,
                 concat=True, aggregator_type="mean", model_size="small",
                 sigmoid_loss=False, identity_dim=0, **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists
                    (padded with random re-samples)
            - degrees: Numpy array with node degrees.
            - layer_infos: List of SAGEInfo namedtuples that describe the
                    parameters of all the recursive layers.
                    See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj

        adj_shape = adj.get_shape().as_list()
        self.loss_node = tf.SparseTensor(
                indices=np.empty((0, 2), dtype=np.int64),
                values=[], dense_shape=[adj_shape[0], adj_shape[0]])
        self.loss_node_count = tf.SparseTensor(
                indices=np.empty((0, 2), dtype=np.int64),
                values=[], dense_shape=[adj_shape[0], adj_shape[0]])

        if identity_dim > 0:
            self.embeds = tf.compat.v1.get_variable(
                    "node_embeddings",
                    [adj.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity " +
                                "feature dimension if no input features given."
                                )
            self.features = self.embeds
        else:
            self.features = tf.Variable(
                    tf.constant(features, dtype=tf.float32), trainable=False)
            if self.embeds is not None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) +
                     identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(
                layer_infos))])

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learning_rate)
        self.build()

    def build(self):
        samples1, support_sizes1, attentions1, num_nz, out_mean = self.sample(
                self.inputs1, self.layer_infos)
        self.att = attentions1
        self.out_mean = out_mean
        num_samples = [layer_info.num_samples for layer_info in
                       self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(
                samples1, [self.features], self.dims, num_samples,
                support_sizes1, attentions1, num_nz, concat=self.concat,
                model_size=self.model_size)
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)

        dim_mult = 2 if self.concat else 1
        self.node_pred = layers.Dense(
                dim_mult*self.dims[-1], self.num_classes, self.weight_decay,
                dropout=self.placeholders['dropout'], act=lambda x : x)

        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)
        self._loss()

        # added
        self.sparse_loss_to_node(samples1, support_sizes1, num_samples)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad
                                   is not None else None, var) for grad, var
                                  in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()

    # added
    def sparse_loss_to_node(self, samples, support_size, num_samples):
        batch_size = self.batch_size
        length = sum(support_size[1:])*batch_size
        node_dim = self.loss_node.get_shape().as_list()

        #discount = .9
        for k in range(1, 2):
            x = tf.reshape(
                    tf.tile(tf.expand_dims(samples[k-1], -1),
                            [1, tf.cast(support_size[k]/support_size[k-1],
                                        tf.int32)]), [-1])
            x = tf.cast(x, tf.int64)
            y = samples[k]
            y = tf.cast(y, tf.int64)
            idx = tf.expand_dims(x*node_dim[0] + y, 1)

            if self.sigmoid_loss is True:
                loss = tf.reshape(tf.tile(tf.expand_dims(tf.reduce_sum(
                        input_tensor=self.cross_entropy+1e-20, axis=1), -1),
                        [1, support_size[k]]), [-1])
            else:
                loss = tf.reshape(tf.tile(tf.expand_dims(
                        self.cross_entropy+1e-20, -1),
                        [1, support_size[k]]), [-1])
            scatter1 = tf.SparseTensor(idx, loss+1e-20, tf.constant(
                    [node_dim[0]*node_dim[1]], dtype=tf.int64))
            scatter1 = tf.sparse.reshape(scatter1, tf.constant(
                    [node_dim[0], node_dim[1]]))
            self.loss_node = tf.sparse.add(a=self.loss_node, b=scatter1)

            ones = tf.reshape(tf.tile(tf.expand_dims(tf.ones(batch_size), -1),
                                      [1, support_size[k]]), [-1])
            scatter2 = tf.SparseTensor(idx, ones, tf.constant(
                    [node_dim[0]*node_dim[1]], dtype=tf.int64))
            scatter2 = tf.sparse.reshape(scatter2, tf.constant(
                    [node_dim[0], node_dim[1]]))
            self.loss_node_count = tf.sparse.add(a=self.loss_node_count,
                                                 b=scatter2)

    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += self.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # classification loss
        if self.sigmoid_loss:
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds, labels=self.placeholders['labels'])
            self.loss += tf.reduce_mean(input_tensor=self.cross_entropy)
        else:
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=tf.stop_gradient(self.placeholders['labels']))
            self.loss += tf.reduce_mean(input_tensor=self.cross_entropy)
        tf.compat.v1.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)
