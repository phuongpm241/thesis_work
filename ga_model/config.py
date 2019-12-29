#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Minibatch Size
BATCH_SIZE = 32
# Gradient clip threshold
GRAD_CLIP = 10
# Learning rate
LEARNING_RATE = 0.0005
# Maximum number of steps in BPTT
GRAD_STEPS = -1
# Number of epochs for training
NUM_EPOCHS = 10
# do validation every VALIDATION_FREQ iterations
VALIDATION_FREQ = 100
# maximum word length for character model
MAX_WORD_LEN = 10
EMBED_SIZE=100

print_every=50  # 1   50
eval_every=500  #100  10000

# dataset params
def get_params(dataset):
    if dataset=='cbtcn':
        return cbtcn_params
    elif dataset=='wdw' or dataset=='wdw_relaxed':
        return wdw_params
    elif dataset=='cnn':
        return cnn_params
    elif dataset=='dailymail':
        return dailymail_params
    elif dataset=='cbtne':
        return cbtne_params
    else:
        raise ValueError("Dataset %s not found"%dataset)

cbtcn_params = {
        'nhidden'   :   128,
        'char_dim'  :   25,
        'dropout'   :   0.4,
        'word2vec'  :   'word2vec_glove.txt',
        'train_emb' :   bool(0),
        'use_feat'  :   bool(1),
        }

wdw_params = {
        'nhidden'   :   128,
        'char_dim'  :   25,
        'dropout'   :   0.3,
        'word2vec'  :   'word2vec_glove.txt',
        'train_emb' :   bool(0),
        'use_feat'  :   bool(1),
        }

cnn_params = {
        'nhidden'   :   256,
        'char_dim'  :   0,
        'dropout'   :   0.2,
        'word2vec'  :   'word2vec_glove.txt',
        'train_emb' :   bool(1),
        'use_feat'  :   bool(0),
        }

dailymail_params = {
        'nhidden'   :   256,
        'char_dim'  :   0,
        'dropout'   :   0.1,
        'word2vec'  :   'word2vec_glove.txt',
        'train_emb' :   bool(1),
        'use_feat'  :   bool(0),
        }

cbtne_params = {
        'nhidden'   :   128,
        'char_dim'  :   25,
        'dropout'   :   0.4,
        'word2vec'  :   'word2vec_glove.txt',
        'train_emb' :   bool(0),
        'use_feat'  :   bool(1),
        }

