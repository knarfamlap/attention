#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from data_loader import load_dataset

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




    
