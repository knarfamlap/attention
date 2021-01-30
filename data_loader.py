#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from lang_idx import LanguageIndex, max_length

def load_dataset(pairs, num_examples):
    """
    pairs: already created and cleaned inputs and output pairs
    """

    inp_lang = LanguageIndex(en for en, other in pairs)
    targ_lang = LanguageIndex(other for en, ma in pairs)


    # Vectorize input and target languages

    # English
    input_tensor = [[inp_lang.word2idx[s] for s in en.split(' ')] for en, ma in pairs]
    # Other Lang
    target_tensor = [[targ_lang.word2idx[s] for s in ma.split(' ')] for en, ma in pairs]
    
    #  Calculate max_length of input and output tensor
    
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

    # Padding the input and output tensor to the max length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen=max_len_inp,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.pad_sequences(target_tensor,
                                                         maxlen=max_length_tar,
                                                         padding='post')


    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar




