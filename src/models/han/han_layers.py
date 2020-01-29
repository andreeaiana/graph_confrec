# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from han_attention import sp_attn_head, SimpleAttLayer

# DISCLAIMER:
# This code file is derived from https://github.com/Jhy1993/HAN.


class GAT_inference(tf.keras.layers.Layer):
    def __init__(self, n_heads, hid_units, nb_classes, nb_nodes, ffd_drop=0.0,
                 attn_drop=0.0, activation=tf.nn.elu, residual=False,
                 return_coef=False):
        super(GAT_inference, self).__init__()

        attned_head = sp_attn_head

        self.attns = []
        self.sec_attns = []
        self.final_attns = []
        self.final_sum = n_heads[-1]

        for i in range(n_heads[0]):
            self.attns.append(attned_head(hidden_dim=hid_units[0],
                                          nb_nodes=nb_nodes,
                                          in_drop=ffd_drop,
                                          coef_drop=attn_drop,
                                          activation=activation,
                                          residual=residual,
                                          return_coef=return_coef))

        for i in range(1, len(hid_units)):
            sec_attns = []
            for j in range(n_heads[i]):
                sec_attns.append(attned_head(hidden_dim=hid_units[i],
                                             nb_nodes=nb_nodes,
                                             in_drop=ffd_drop,
                                             coef_drop=attn_drop,
                                             activation=activation,
                                             residual=residual,
                                             return_coef=return_coef))
                self.sec_attns.append(sec_attns)

        for i in range(n_heads[-1]):
            self.final_attns.append(attned_head(hidden_dim=nb_classes,
                                                nb_nodes=nb_nodes,
                                                in_drop=ffd_drop,
                                                coef_drop=attn_drop,
                                                activation=lambda x: x,
                                                residual=residual,
                                                return_coef=return_coef))

    def __call__(self, inputs, bias_mat, training):
        first_attn = []
        out = []

        for indiv_attn in self.attns:
            first_attn.append(indiv_attn(seq=inputs, bias_mat=bias_mat,
                                         training=training))
        h_1 = tf.concat(first_attn, axis=-1)

        for sec_attns in self.sec_attns:
            next_attn = []
            for indiv_attns in sec_attns:
                next_attn.append(indiv_attns(seq=h_1, bias_mat=bias_mat,
                                             training=training))
            h_1 = tf.concat(next_attn, axis=-1)

        for indiv_attn in self.final_attns:
            out.append(indiv_attn(seq=h_1, bias_mat=bias_mat,
                                  training=training))

        logits = tf.add_n(out)/self.final_sum
        return logits


class HeteGAT_multi_inference(tf.keras.layers.Layer):
    def __init__(self,  n_heads, hid_units, nb_classes, nb_nodes,
                 ffd_drop=0.0, attn_drop=0.0, activation=tf.nn.elu,
                 residual=False, return_coef=False, mp_att_size=128):
        super(HeteGAT_multi_inference, self).__init__()
        attned_head = sp_attn_head
        self.attns = []
        self.sec_attns = []
        self.final_attns = []
        self.final_sum = n_heads[-1]
        for i in range(n_heads[0]):
            self.attns.append(attned_head(hidden_dim=hid_units[0],
                                          nb_nodes=nb_nodes,
                                          in_drop=ffd_drop,
                                          coef_drop=attn_drop,
                                          activation=activation,
                                          residual=residual,
                                          return_coef=return_coef))

        for i in range(1, len(hid_units)):
            sec_attns = []
            for j in range(n_heads[i]):
                sec_attns.append(attned_head(hidden_dim=hid_units[i],
                                             nb_nodes=nb_nodes,
                                             in_drop=ffd_drop,
                                             coef_drop=attn_drop,
                                             activation=activation,
                                             residual=residual,
                                             return_coef=return_coef))
                self.sec_attns.append(sec_attns)

        for i in range(n_heads[-1]):
            self.final_attns.append(tf.keras.layers.Dense(nb_classes,
                                                          activation=None))

        self.simple_attn_layer = SimpleAttLayer(attention_size=mp_att_size,
                                                time_major=False,
                                                return_alphas=True)

    def __call__(self, inputs_list, bias_mat_list, training):
        embed_list = []
        out = []
        for inputs, bias_mat in zip(inputs_list, bias_mat_list):
            first_attn = []
            embeds = []
            i = 1
            for indiv_attn in self.attns:
                first_attn.append(indiv_attn(seq=inputs,
                                             bias_mat=bias_mat,
                                             training=training))
                i += 1
            h_1 = tf.concat(first_attn, axis=-1)

            for sec_attns in self.sec_attns:
                next_attn = []
                for indiv_attns in sec_attns:
                    next_attn.append(indiv_attns(seq=h_1,
                                                 bias_mat=bias_mat,
                                                 training=training))
                h_1 = tf.concat(next_attn, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))
        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = self.simple_attn_layer(multi_embed)

        for indiv_attn in self.final_attns:
            out.append(indiv_attn(final_embed))

        logits = tf.add_n(out) / self.final_sum
        logits = tf.expand_dims(logits, axis=0)
        return logits, final_embed, att_val


class HeteGAT_inference(tf.keras.layers.Layer):
    def __init__(self, n_heads, hid_units, nb_classes, nb_nodes,
                 ffd_drop=0.0, attn_drop=0.0, activation=tf.nn.elu,
                 residual=False, return_coef=False, mp_att_size=128):
        super(HeteGAT_inference, self).__init__()
        attned_head = sp_attn_head
        self.attns = []
        self.head_coefs = []
        self.sec_attns = []
        self.final_attns = []
        self.final_sum = n_heads[-1]

        for i in range(n_heads[0]):
            if return_coef:
                a1, a2 = attned_head(hidden_dim=hid_units[0],
                                     nb_nodes=nb_nodes, in_drop=ffd_drop,
                                     coef_drop=attn_drop,
                                     activation=activation, residual=residual,
                                     return_coef=return_coef)
                self.attns.append(a1)
                self.head_coefs.append(a2)
            else:
                self.attns.append(attned_head(hidden_dim=hid_units[0],
                                              nb_nodes=nb_nodes,
                                              in_drop=ffd_drop,
                                              coef_drop=attn_drop,
                                              activation=activation,
                                              residual=residual,
                                              return_coef=return_coef))
        for i in range(1, len(hid_units)):
            sec_attns = []
            for j in range(n_heads[i]):
                sec_attns.append(attned_head(hidden_dim=hid_units[i],
                                             nb_nodes=nb_nodes,
                                             in_drop=ffd_drop,
                                             coef_drop=attn_drop,
                                             activation=activation,
                                             residual=residual,
                                             return_coef=return_coef))
                self.sec_attns.append(sec_attns)
        for i in range(n_heads[-1]):
            self.final_attns.append(tf.keras.layers.Dense(nb_classes,
                                                          activation=None))
        self.simple_attn_layer = SimpleAttLayer(attention_size=mp_att_size,
                                                time_major=False,
                                                return_alphas=True)
        self.return_coef = return_coef

    def __call__(self, inputs, bias_mat_list, training):
        embed_list = []
        coef_list = []
        out = []
        for bias_mat in bias_mat_list:
            first_attn = []
            head_coef_list = []
            if self.return_coef:
                for indiv_attn in self.attns:
                    first_attn.append(indiv_attn(seq=inputs,
                                                 bias_mat=bias_mat,
                                                 training=training))
                for h_c in self.head_coefs:
                    head_coef_list.append(h_c(seq=inputs,
                                              bias_mat=bias_mat,
                                              training=training))
            else:
                for indiv_attn in self.attns:
                    first_attn.append(indiv_attn(seq=inputs,
                                                 bias_mat=bias_mat,
                                                 training=training))
            if self.return_coef:
                head_coef = tf.concat(head_coef_list, axis=0)
                head_coef = tf.reduce_mean(input_tensor=head_coef, axis=0)
                coef_list.append(head_coef)
            h_1 = tf.concat(first_attn, axis=-1)

            for sec_attns in self.sec_attns:
                next_attn = []
                for indiv_attns in sec_attns:
                    next_attn.append(indiv_attns(seq=h_1,
                                                 bias_mat=bias_mat,
                                                 training=training))
                h_1 = tf.concat(next_attn, axis=-1)
            embed_list.append(tf.expand_dims(tf.squeeze(h_1), axis=1))

        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, attn_val = self.simple_attn_layer(multi_embed)

        for indiv_attn in self.final_attns:
            out.append(indiv_attn(final_embed))

        logits = tf.add_n(out) / self.final_sum
        logits = tf.expand_dims(logits, axis=0)
        if self.return_coef:
            return logits, final_embed, attn_val, coef_list
        else:
            return logits, final_embed, attn_val
