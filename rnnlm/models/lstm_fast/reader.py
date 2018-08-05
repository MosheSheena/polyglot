

"""Utilities for parsing RNNLM text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf


def _int64_feature_wrap(int_values):
    """
    This wraps tf.train.feature.
    This function in used in the process of writing tf records
    Args:
        int_values: (list) a list of integers

    Returns:
        (tf.train.Feature)
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int_values))


def write_tf_records(word_indices, destination_path):
    """
    Writes the data in a tf record format
    Args:
        word_indices: (list)
        destination_path: (str) where to write the tf records files
    Returns:

    """
    word_feature = _int64_feature_wrap(int_values=word_indices)
    feature_dict = {
        "words": word_feature
    }
    words = tf.train.Features(feature=feature_dict)

    example = tf.train.Example(features=words)

    with tf.python_io.TFRecordWriter(destination_path) as writer:
        writer.write(example.SerializeToString())
        

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        # return f.read().decode("utf-8").split() # use this if data not in english
        return f.read().split()


def _build_vocab(filename):
    words = _read_words(filename)
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
