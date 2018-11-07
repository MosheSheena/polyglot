"""Utilities for parsing RNNLM files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from rnnlm.utils.tf_io.generator import _gen_read_n_shifted_elements, READ_ENTIRE_FILE_MODE


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


def read_tf_records(abs_tf_record_path, batch_size, seq_len, dtype_features, dtype_labels, shuffle=False, skip_first_n=0):
    """
    reads for set of tf record files into a data set
    Args:
        abs_tf_record_path (str): absolute path to load the tf record file from
        batch_size (int):
        seq_len (int):
        dtype_features (tf.DType): should match to what was wrote to tf records
        dtype_labels(tf.DType): should match to what was wrote to tf records
        shuffle (bool): whether to shuffle or not
        skip_first_n (int): Optional num of records to skip at the beginning of the dataset
            default is 0
    Returns:
        tf.data.TFRecordDataset, an object representing our dataset.
    """

    dataset = tf.data.TFRecordDataset(abs_tf_record_path)
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


def read_and_build_vocab(file_obj):
    """
    Create a mapping for each element of the vocab to its ID.
    Args:
        file_obj: Opened vocab file

    Returns:
        dict that maps elements to its ID
    """
    gen_elements = _gen_read_n_shifted_elements(file_obj=file_obj, n=READ_ENTIRE_FILE_MODE)
    elements = next(gen_elements)
    element_to_id = dict(zip(elements, range(len(elements))))
    return element_to_id
