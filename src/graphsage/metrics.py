import tensorflow as tf

# DISCLAIMER:
# This code file is forked  from https://github.com/williamleif/GraphSAGE,
# which is under an identical MIT license as graph_confrec.
# Parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package


def masked_logit_cross_entropy(preds, labels, mask):
    """Logit cross-entropy loss with masking."""
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    loss = tf.reduce_sum(input_tensor=loss, axis=1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.maximum(tf.reduce_sum(input_tensor=mask), tf.constant([1.]))
    loss *= mask
    return tf.reduce_mean(input_tensor=loss)


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=preds, labels=tf.stop_gradient(labels))
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.maximum(tf.reduce_sum(input_tensor=mask), tf.constant([1.]))
    loss *= mask
    return tf.reduce_mean(input_tensor=loss)


def masked_l2(preds, actuals, mask):
    """L2 loss with masking."""
    loss = tf.nn.l2(preds, actuals)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    loss *= mask
    return tf.reduce_mean(input_tensor=loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(input=preds, axis=1),
                                  tf.argmax(input=labels, axis=1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    accuracy_all *= mask
    return tf.reduce_mean(input_tensor=accuracy_all)
