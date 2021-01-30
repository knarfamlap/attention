#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

class Decoder(tf.keras.Model):
    def __init__(self, vocab_sz, embedding_dim, decoder_units, batch_sz):
        super(Decoder, self).__init__()

        self.batch_sz = batch_sz
        self.decoder_units = decoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.decoder_units,
                                       return_sequences=True, 
                                       recurrent_activation='sigmoid',
                                       recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_sz) 

        self.W1 = tf.keras.layers.Dense(self.decoder_units)
        self.W2 = tf.keras.layers.Dense(self.decoder_units)
        self.V = tf.keras.layers.Dense(1)


