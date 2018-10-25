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


def gen_shifted_words_with_overlap(file_obj, seq_len):

    gen_words = _read_n_shifted_words_gen(file_obj=file_obj, n=seq_len, overlap=True)
    x = next(gen_words)
    y = next(gen_words)
    while True:
        try:
            if x[1:] != y[:-1] or len(x) != len(y):
                raise StopIteration  # ignore remainder that is less than the sequence length

            yield (x, y)
            # x = y since the words are shifted by 1 in time
            x = y
            y = next(gen_words)

        except StopIteration:
            break


def gen_no_overlap_words(file_obj, seq_len):

    gen_words = _read_n_shifted_words_gen(file_obj=file_obj, n=seq_len, overlap=False)

    accumulator = next(gen_words)
    while True:
        try:
            x = accumulator
            accumulator = next(gen_words)
            y = x[1:] + [accumulator[0]]

            if x[1:] != y[:-1] or len(x) != len(y):
                raise StopIteration  # ignore remainder that is less than the sequence length

            yield (x, y)

        except StopIteration:
            break


def gen_pos_tagger(file_obj, seq_len, overlap=False):
    """
    Each call to this generator generates a tuple of x, y
    where x is list of words with a list size of seq_len
    and y is a list of part-of-speech tags of the words
    in x accordingly.
    Args:
        file_obj: opened file of raw data
        seq_len: the length of how much we will read from the
            file in each generation
        overlap: whether to generate with overlaps or not

    Returns:

    """

    gen_words = _read_n_shifted_words_gen(file_obj=file_obj, n=seq_len, overlap=overlap)

        # yield x, y


def _parse_fn(example_proto, seq_len, dtype_features, dtype_labels):
    """
    Parses a single example from tf record files into Tensor or SparseTensor

    Args:
        example_proto: (tf.train.Example) example from tf record file
        seq_len (int):
        dtype_features (tf.DType): should match to what was wrote to tf records
        dtype_labels (tf.DType): should match to what was wrote to tf records

    Returns:
        if tf.FixedLenFeature -> return Tensor
        if tf.VarLenFeature -> return SparseTensor
    """
    read_features = {
        "x": tf.FixedLenFeature(shape=[seq_len], dtype=dtype_features),
        "y": tf.FixedLenFeature(shape=[seq_len], dtype=dtype_labels),
    }
    parsed_features = tf.parse_single_example(example_proto, read_features)
    return parsed_features["x"], parsed_features["y"]


def read_tf_records(tf_record_path, batch_size, seq_len, dtype_features, dtype_labels, shuffle=False, skip_first_n=0):
    """
    reads for set of tf record files into a data set
    Args:
        tf_record_path (str): where to load the tf record file from
        batch_size (int):
        seq_len (int):
        dtype_features (tf.DType): should match to what was wrote to tf records
        dtype_labels(tf.DType): should match to what was wrote to tf records
        shuffle (bool):
        skip_first_n (int): Optional num of records to skip at the beginning of the dataset
            default is 0
    Returns:
        next_op for one shot iterator of the dataset, where each session run op on
        the returned tensor yields x, y with shape [batch_size, seq_len]
    """

    dataset = tf.data.TFRecordDataset(tf_record_path)
    dataset = dataset.map(
        lambda x: _parse_fn(x, seq_len, dtype_features, dtype_labels),
        num_parallel_calls=4
    )  # parse into tensors
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.skip(count=skip_first_n)
    dataset = dataset.prefetch(buffer_size=1)
    return dataset


def build_vocab(file_obj):
    gen_words = _read_n_shifted_words_gen(file_obj, READ_ENTIRE_FILE_MODE)
    words = next(gen_words)
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id
