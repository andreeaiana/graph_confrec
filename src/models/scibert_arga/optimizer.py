import tensorflow as tf

# DISCLAIMER:
# This code file is derived from https://github.com/Ruiqi-Hu/ARGA,
# which is under an identical MIT license as graph_confrec.


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm, d_real, d_fake,
                 discriminator_learning_rate, learning_rate):
        preds_sub = preds
        labels_sub = labels

        self.real = d_real
        self.discriminator_learning_rate = discriminator_learning_rate
        self.learning_rate = learning_rate

        # Discrimminator Loss
        self.dc_loss_real = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(self.real), logits=self.real,
                    name='dclreal'))

        self.dc_loss_fake = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(d_fake), logits=d_fake,
                    name='dcfake'))
        self.dc_loss = self.dc_loss_fake + self.dc_loss_real

        # Generator loss
        generator_loss = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(d_fake), logits=d_fake, name='gl'))

        self.cost = norm * tf.reduce_mean(
                input_tensor=tf.nn.weighted_cross_entropy_with_logits(
                        logits=preds_sub, labels=labels_sub,
                        pos_weight=pos_weight))
        self.generator_loss = generator_loss + self.cost
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learning_rate)  # Adam Optimizer

        all_variables = tf.compat.v1.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.name]
        en_var = [var for var in all_variables if 'e_' in var.name]

        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            self.discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.discriminator_learning_rate, beta1=0.9,
                    name='adam1').minimize(self.dc_loss, var_list=dc_var)
            self.generator_optimizer = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.discriminator_learning_rate, beta1=0.9,
                    name='adam2').minimize(self.generator_loss,
                                           var_list=en_var)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm,
                 d_real, d_fake, discriminator_learning_rate, learning_rate):
        preds_sub = preds
        labels_sub = labels
        self.discriminator_learning_rate = discriminator_learning_rate
        self.learning_rate = learning_rate

        # Discrimminator Loss
        dc_loss_real = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(d_real), logits=d_real))
        dc_loss_fake = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(d_fake), logits=d_fake))
        self.dc_loss = dc_loss_fake + dc_loss_real

        # Generator loss
        self.generator_loss = tf.reduce_mean(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(d_fake), logits=d_fake))

        self.cost = norm * tf.reduce_mean(
            input_tensor=tf.nn.weighted_cross_entropy_with_logits(
                    logits=preds_sub, labels=labels_sub, pos_weight=pos_weight)
            )
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learning_rate)  # Adam Optimizer

        all_variables = tf.compat.v1.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.op.name]
        en_var = [var for var in all_variables if 'e_' in var.op.name]

        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(),
                                         reuse=False):
            self.discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.discriminator_learning_rate, beta1=0.9,
                    name='adam1').minimize(self.dc_loss, var_list=dc_var)

            self.generator_optimizer = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.discriminator_learning_rate, beta1=0.9,
                    name='adam2').minimize(self.generator_loss, var_list=en_var
                                           )

        self.cost = norm * tf.reduce_mean(
                input_tensor=tf.nn.weighted_cross_entropy_with_logits(
                        logits=preds_sub, labels=labels_sub,
                        pos_weight=pos_weight))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learning_rate)  # Adam Optimizer
        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(
                input_tensor=tf.reduce_sum(
                        input_tensor=1 + 2 * model.z_log_std - tf.square(
                                model.z_mean) - tf.square(tf.exp(
                                        model.z_log_std)), axis=1))
        self.cost -= self.kl
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
