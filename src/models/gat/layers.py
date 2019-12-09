# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from attention import attn_head, sp_attn_head

# DISCLAIMER:
# This code file is derived from https://github.com/PetarV-/GAT,
# which is under an identical MIT license as graph_confrec.
# Conversion of original code to TF 2.0 is inspired by
# https://github.com/calciver/Graph-Attention-Networks/blob/master/Tensorflow_2_0_Graph_Attention_Networks_(GAT).ipynb


class inference(tf.keras.layers.Layer):
    def __init__(self, n_heads, hid_units, nb_classes, nb_nodes, Sparse,
                 ffd_drop=0.0, attn_drop=0.0, activation=tf.nn.elu,
                 residual=False):
        super(inference, self).__init__()

        attned_head = self._choose_attn_head(Sparse)

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
                                          residual=residual))

        for i in range(1, len(hid_units)):
            sec_attns = []
            for j in range(n_heads[i]):
                sec_attns.append(attned_head(hidden_dim=hid_units[i],
                                             nb_nodes=nb_nodes,
                                             in_drop=ffd_drop,
                                             coef_drop=attn_drop,
                                             activation=activation,
                                             residual=residual))
                self.sec_attns.append(sec_attns)

        for i in range(n_heads[-1]):
            self.final_attns.append(attned_head(hidden_dim=nb_classes,
                                                nb_nodes=nb_nodes,
                                                in_drop=ffd_drop,
                                                coef_drop=attn_drop,
                                                activation=lambda x: x,
                                                residual=residual))

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
                next_attn.append(indiv_attn(seq=h_1, bias_mat=bias_mat,
                                            training=training))
            h_1 = tf.concat(next_attns, axis=-1)

        for indiv_attn in self.final_attns:
            out.append(indiv_attn(seq=h_1, bias_mat=bias_mat,
                                  training=training))

        logits = tf.add_n(out)/self.final_sum
        return logits

    def _choose_attn_head(self, Sparse):
        if Sparse:
            chosen_attention = sp_attn_head
        else:
            chosen_attention = attn_head
        return chosen_attention
