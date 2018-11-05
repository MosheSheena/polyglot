import tensorflow as tf
from rnnlm.models.lstm_fast import reader


def map_elements_to_ids(elements, vocab):

    return [vocab[element] if element in vocab else vocab["<oos>"] for element in elements]


def build_vocab(vocab_file_path):
    with tf.gfile.GFile(vocab_file_path, 'r') as vocab_file:
        vocab = reader.build_vocab(vocab_file)
    packed_vocab = [vocab]
    return packed_vocab
