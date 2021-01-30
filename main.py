#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import time
from data_loader import load_dataset
from utils import predict_random_val_sentence
from decoder import Decoder
from encoder import Encoder

input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar = load_dataset(sent_pairs, len(lines))
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, 
                                                                                                target_tensor,
                                                                                                test_size=0.1, 
                                                                                                random_state=42)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE) 

optimizer = tf.train.AdamOptimizer()

def loss_fn(real, pred):
    mask = 1 - np.equal(real, 0) 
    loss_ = tf.nn.sparse_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

checkpoint_dir = './checkpoints'
checkpoints_prefix = os.path.join(checkpoint_dir, 'ckpt') 
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                encoder=encoder, 
                                decoder=decoder) 

EPOCHS = 10

for epoch in range(EPOCHS):

    start = time.time()

    hidden = encoder.init_hidden()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset):
        loss = 0

        with tf.GradientTape as tape:
            enc_output, enc_hidden = encoder(inp, hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1) 

            for t in range(1, targ.shape[1]):
                predicitons, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += loss_fn(targ[:, t], predicitons)

                dec_input = tf.expand_dims(targ[:, t], 1) 



        batch_loss = (loss / int(targ.shape[1])) 

        total_loss += batch_loss

        variables = encoder.variables + decoder.variables

        gradients = tape.gradients(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, 
                                                         batch, 
                                                         batch_loss.numpy())) 


    checkpoints.save(file_prefix = checkpoints_prefix)
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / N_BATCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


predict_random_val_sentence() 
