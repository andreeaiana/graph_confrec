# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from layers import inference

# DISCLAIMER:
# This code file is derived from https://github.com/PetarV-/GAT,
# which is under an identical MIT license as graph_confrec.
# Conversion of original code to TF 2.0 is inspired by
# https://github.com/calciver/Graph-Attention-Networks/blob/master/Tensorflow_2_0_Graph_Attention_Networks_(GAT).ipynb


class GAT(tf.keras.Model):
    def __init__(self, hid_units, n_heads, nb_classes, nb_nodes, Sparse,
                 l2_coef=0.0005, ffd_drop=0.0, attn_drop=0.0,
                 activation=tf.nn.elu, residual=False):
        """
        hid_units: The number of hidden units per each attention head
                    in each layer. Array of hidden layer dimensions.
        n_heads: This is the additional entry of the output layer.
                More specifically the output that calculates attn
        nb_classes: This refers to the number of classes (7)
        nb_nodes: This refers to the number of nodes (2708)
        activation: This is the activation function tf.nn.elu
        residual: This determines whether we add seq to ret (False)
        """
        super(GAT, self).__init__()

        self.nb_classes = nb_classes
        self.l2_coef = l2_coef
        self.inferencing = inference(
                n_heads, hid_units, self.nb_classes, nb_nodes, Sparse=Sparse,
                ffd_drop=ffd_drop, attn_drop=attn_drop, activation=activation,
                residual=residual)

    ##########################
    # Adapted from tkipf/gcn #
    ##########################

    def masked_softmax_cross_entropy(self, logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
#        loss = tf.nn.softmax_cross_entropy_with_logits(
#                logits=logits, labels=tf.stop_gradient(labels))
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_sigmoid_cross_entropy(self, logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels)
        loss = tf.reduce_mean(loss, axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(self, logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(logits, axis=1),
                                      tf.argmax(labels, axis=1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def micro_f1(self, logits, labels, mask):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))

        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)

        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(mask, -1)

        # Count true positives, true negatives, false positives and false
        # negatives.
        tp = tf.count_nonzero(predicted * labels * mask)
        tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
        fp = tf.count_nonzero(predicted * (labels - 1) * mask)
        fn = tf.count_nonzero((predicted - 1) * labels * mask)

        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure

    def __call__(self, inputs, training, bias_mat, lbl_in, msk_in):
        logits = self.inferencing(inputs=inputs, bias_mat=bias_mat,
                                  training=training)

        log_resh = tf.reshape(logits, [-1, self.nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, self.nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])

        loss = self.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables
                           if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']
                           ]) * self.l2_coef
        loss = loss+lossL2
        accuracy = self.masked_accuracy(log_resh, lab_resh, msk_resh)

        return logits, accuracy, loss
