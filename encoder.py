#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_sz,  embedding_dim, encoder_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.encoder_units = encoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_sz, embedding_dim)
        self.gru = tf .keras.layers.GRU(self.encoder_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_activation='sigmoid',
                                        recurrent_initializer='glorot_uniform')

    def __call__(self, x, hidden):
        x = self.Embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def init_hidden(self):
        return tf.zeros((self.batch_sz, self.encoder_units))




