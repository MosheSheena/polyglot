"""Utilities for parsing RNNLM text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

READ_ENTIRE_FILE_MODE = -1


def read_n_shifted_words_gen(file_obj, n):
    """
    Generator function that reads n words each time from file.
    Each yield contains a list of words shifted by 1
    if n = 0 it returns an empty list
    if n is negative it returns all the words from file_obj
    Args:
        file_obj: opened file
        n: (int) num of words to read from file

    Returns:
        list of n words from file
    """
    if n < 0:
        yield list(file_obj)
    elif n == 0:
        yield list()
    else:
        n_words = list()

        for line in file_obj:
            for word in line.split():
                n_words.append(word)
                if len(n_words) == n:
                    yield list(n_words)
                    # remove the first element
                    # from here and on one element will be inserted to the list and will be yield
                    n_words.pop(0)

        # take care of the remainder of num_words % n
        if len(n_words) % n != 0:
            yield n_words


def _parse_fn(example_proto):
    """
    Parses a single example from tf record files into Tensor or SparseTensor

    Args:
        example_proto: (tf.train.Example) example from tf record file

    Returns:
        if tf.FixedLenFeature -> return a Tensor
        if tf.VarLenFeature -> return SparseTensor
    """
    read_features = {
        "x": tf.VarLenFeature(dtype=tf.int64),
        "y": tf.VarLenFeature(dtype=tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, read_features)
    return parsed_features["x"], parsed_features["y"]


def read_tf_records():
    """
    reads for set of tf record files into a data set
    Args:

    Returns:

    """

    # use a tensor for file names - better when using different data for validation and test sets
    tf_record_paths = tf.placeholder(tf.string, shape=[None], name="tf_record_paths")
    dataset = tf.data.TFRecordDataset(tf_record_paths)
    dataset = dataset.map(_parse_fn)  # parse into tensors


def build_vocab(file_obj):
    gen_words = read_n_shifted_words_gen(file_obj, READ_ENTIRE_FILE_MODE)
    words = next(gen_words)
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def rnnlm_raw_data(data_path, vocab_path):
    """Load RNNLM raw data from data directory "data_path".

    Args:
      data_path: string path to the directory where train/valid/test files are stored
      vocab_path: string path to the directory where the vocabulary is stored

    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to RNNLMIterator.
    """

    train_path = os.path.join(data_path, "train")
    valid_path = os.path.join(data_path, "valid")
    test_path = os.path.join(data_path, "test")

    word_to_id = _build_vocab(vocab_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary_len = len(word_to_id)
    
    return train_data, valid_data, test_data, vocabulary_len, word_to_id, word_to_id


def rnnlm_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw RNNLM data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
      raw_data: one of the raw data outputs from rnnlm_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).

    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.

    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "RNNLMProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps],
                             [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y
