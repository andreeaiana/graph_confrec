# -*- coding: utf-8 -*-
import tensorflow as tf
from han_layers import HeteGAT_inference, HeteGAT_multi_inference

# DISCLAIMER:
# This code file is derived from https://github.com/Jhy1993/HAN.


class HAN(tf.keras.Model):
    def __init__(self, layer, hid_units, n_heads, nb_classes, nb_nodes,
                 l2_coef=0.0005, ffd_drop=0.0, attn_drop=0.0,
                 activation=tf.nn.elu, residual=False):
        super(HAN, self).__init__()

        self.nb_classes = nb_classes
        self.l2_coef = l2_coef
        self.layer = layer
        if self.layer == "HeteGAT":
            self.inferencing = HeteGAT_inference(
                        n_heads, hid_units, self.nb_classes, nb_nodes,
                        ffd_drop=ffd_drop, attn_drop=attn_drop,
                        activation=activation, residual=residual,
                        return_coef=False, mp_att_size=128)
        elif self.layer == "HeteGAT_multi":
            self.inferencing = HeteGAT_multi_inference(
                    n_heads, hid_units, self.nb_classes, nb_nodes,
                    ffd_drop=ffd_drop, attn_drop=attn_drop,
                    activation=activation, residual=residual,
                    return_coef=False, mp_att_size=128)
        else:
            raise ValueError("Layer name not recognised.")

    def preshape(self, logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(self, logits, labels):
        preds = tf.argmax(input=logits, axis=1)
        return tf.math.confusion_matrix(labels=labels, predictions=preds)

    ##########################
    # Adapted from tkipf/gcn #
    ##########################

    def masked_softmax_cross_entropy(self, logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(input_tensor=mask)
        loss *= mask
        return tf.reduce_mean(input_tensor=loss)

    def masked_sigmoid_cross_entropy(self, logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labels)
        loss = tf.reduce_mean(input_tensor=loss, axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(input_tensor=mask)
        loss *= mask
        return tf.reduce_mean(input_tensor=loss)

    def masked_accuracy(self, logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(input=logits, axis=1),
                                      tf.argmax(input=labels, axis=1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(input_tensor=mask)
        accuracy_all *= mask
        return tf.reduce_mean(input_tensor=accuracy_all)

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

    def __call__(self, inputs_list, training, bias_mat_list, lbl_in, msk_in):
        if self.layer == "HeteGAT":
            logits, embed, att_val = self.inferencing(
                    inputs=inputs_list[0], bias_mat_list=bias_mat_list,
                    training=training)
        else:
            logits, embed, att_val = self.inferencing(
                    inputs_list=inputs_list, bias_mat_list=bias_mat_list,
                    training=training)
        log_resh = tf.reshape(logits, [-1, self.nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, self.nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])
        loss = self.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables
                           if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']
                           ]) * self.l2_coef
        loss = loss + lossL2
        accuracy = self.masked_accuracy(log_resh, lab_resh, msk_resh)

        return logits, embed, att_val, accuracy, loss

