"""Utilities for parsing RNNLM files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

READ_ENTIRE_FILE_MODE = -1


def _read_n_shifted_words_gen(file_obj, n, overlap=False):
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
        yield list(file_obj.read().split())
    elif n == 0:
        yield list()
    else:
        n_words = list()

        for line in file_obj:
            for word in line.split():
                n_words.append(word)
                if len(n_words) == n:
                    yield list(n_words)

                    if overlap:
                        # remove the first element
                        # from here and on one element will be inserted to the list and will be yield
                        n_words.pop(0)
                    else:
                        # flush all elements and get n new ones
                        n_words.clear()

        # take care of the remainder of num_words % n
        if len(n_words) % n != 0:
            yield n_words


def gen_shifted_word(file_obj, seq_len):
    gen_words = _read_n_shifted_words_gen(file_obj=file_obj, n=seq_len, overlap=True)
    x = next(gen_words)
    y = next(gen_words)
    while True:
        try:
            if x[1:] != y[:-1] and len(x) != len(y):
                raise StopIteration  # ignore remainder that is less than the sequence length
            # check that y equals x shifted by 1
            assert (x[1:] == y[:-1] and len(x) == len(y)), "x ={}\ny={}\n".format(x[1:], y[:-1])
            yield (x, y)
            # x = y since the words are shifted by 1 in time
            x = y
            y = next(gen_words)
        except StopIteration:
            break


def _parse_fn(example_proto, seq_len):
    """
    Parses a single example from tf record files into Tensor or SparseTensor

    Args:
        example_proto: (tf.train.Example) example from tf record file

    Returns:
        if tf.FixedLenFeature -> return Tensor
        if tf.VarLenFeature -> return SparseTensor
    """
    read_features = {
        "x": tf.FixedLenFeature(shape=[seq_len], dtype=tf.int64),
        "y": tf.FixedLenFeature(shape=[seq_len], dtype=tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, read_features)
    return parsed_features["x"], parsed_features["y"]


def read_tf_records(tf_record_path, batch_size, seq_len):
    """
    reads for set of tf record files into a data set
    Args:
        tf_record_path: (str) where to load the tf record file from
        batch_size: (int)
        seq_len: (int)
    Returns:
        next_op for one shot iterator of the dataset
    """

    # TODO - possible performance enhancement - check tf.data tutorial
    dataset = tf.data.TFRecordDataset(tf_record_path)
    dataset = dataset.map(lambda x: _parse_fn(x, seq_len))  # parse into tensors
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    return dataset.make_one_shot_iterator().get_next()


def build_vocab(file_obj):
    gen_words = _read_n_shifted_words_gen(file_obj, READ_ENTIRE_FILE_MODE)
    words = next(gen_words)
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


# TODO - remove this method when we are sure tf.data has replaced this correctly
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
