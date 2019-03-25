# -*- coding: utf-8 -*-
"""
A config file containing definitions required for generating words
"""
# Tensors required to load from model
TENSORS_OF_MODEL_DICT = {
    "word_in": "Train/Model/test_word_in",
    "word_out": "Train/Model/test_word_out",
    "initial_state": "Train/Model/test_initial_state",
    "state_in": "Train/Model/test_state_in",
    "state_out": "Train/Model/test_state_out",
    "cell_in": "Train/Model/test_cell_in",
    "cell_out": "Train/Model/test_cell_out",
    "test_out": "Train/Model/test_out",
    "softmax_w": "Model/lstm_fast_softmax/softmax_w",
    "softmax_b": "Model/lstm_fast_softmax/softmax_b"
}

GEN_SENTENCE = 1
GEN_SENTENCES = 2
EXIT = 3

UNKNOWN = '<unk>'
START = '</s>'
OUT_OF_SCOPE = '<oos>'

DEFUALT_SENTENCE_LENGTH = 5
DEFUALT_NUM_WORDS_FOR_FILE = 10
DEFAULT_TEMPERATURE = 0.1

