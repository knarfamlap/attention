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
        # attention weights
        self.W1 = tf.keras.layers.Dense(self.decoder_units)
        self.W2 = tf.keras.layers.Dense(self.decoder_units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, x, hidden, encoder_output):
        """
        enc_output: shape of (batch_sz, max_length, hidden_size)
        hidden: shape of (batch_size, hidden_size)
        """

        hidden_with_times_axis = tf.expand_dims(hidden, 1) # shape of (batch_sz, 1, hidden_sz)

        score = self.V(tf.nn.tanh(self.W1(encoder_output) + self.W2(hidden_with_times_axis)))

        attention_weights = tf.nn.softmax(score, axis=1) # computes e1, e2, ...
        # shape of (batch_sz, hidden_sz)
        context_vector = attention_weights * encoder_output # e1 * h1 + e2 * h2 .. 
        context_vector = tf.reduce_sum(context_vector, axis=1)

        x = self.embedding(x) # x after passed shape of (batch_sz, 1, embedding_dim)

        x = tf.concat([tf.expand_dims(context_vector, 1), x ], axis=1)

        output, state = self.gru(x) 
        # shape (batch_sz , 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output) 

        return x, state, attention_weights

    def init_hidden(self):
        return tf.zeros((self.batch_sz, self.decoder_units))





