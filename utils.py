#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Set the file path
file_path = './mar.txt' # this might be different in your system

# read the file
lines = open(file_path, encoding='UTF-8').read().strip().split('\n')

# perform basic cleaning
exclude = set(string.punctuation) # Set of all special characters
remove_digits = str.maketrans('', '', string.digits) # Set of all digits

def preprocess_eng_sentence(sent):
    '''Function to preprocess English sentence'''
    sent = sent.lower()
    sent = re.sub("'", '', sent)
    sent = ''.join(ch for ch in sent if ch not in exclude)
    sent = sent.translate(remove_digits)
    sent = sent.strip()
    sent = re.sub(" +", " ", sent)
    sent = '<start> ' + sent + ' <end>'


def preprocess_other_lang_sentence(sent):
    '''Function to preprocess Marathi sentence'''
    sent = re.sub("'", '', sent)
    sent = ''.join(ch for ch in sent if ch not in exclude)
    sent = sent.strip()
    sent = re.sub(" +", " ", sent)
    sent = '<start> ' + sent + ' <end>'
    return sent

def generate(lines):
    sent_pairs = []
    for line in lines:
        sent_pair = []
        eng, other = line.split('\t')

        eng = preprocess_eng_sentence(eng)
        other = preprocess_other_lang_sentence(other)

        sent_pair.append(eng)
        sent_pair.append(other)
        
        sent_pairs.append(sent_pair)

    return sent_pairs 
