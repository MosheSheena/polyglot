import tensorflow as tf


def _parse_fn(example_proto,
              feature_sample_size,
              label_sample_size,
              dtype_features,
              dtype_labels):
    """
    Parses a single example from tf record files into Tensor or SparseTensor

    Args:
        example_proto: (tf.train.Example) example from tf record file
        feature_sample_size (int):
        label_sample_size (int):
        dtype_features (tf.DType): should match to what was wrote to tf records
        dtype_labels (tf.DType): should match to what was wrote to tf records

    Returns:
        if tf.FixedLenFeature -> return Tensor
        if tf.VarLenFeature -> return SparseTensor
    """
    read_features = {
        "x": tf.FixedLenFeature(shape=[feature_sample_size], dtype=dtype_features),
        "y": tf.FixedLenFeature(shape=[label_sample_size], dtype=dtype_labels),
    }
    parsed_features = tf.parse_single_example(example_proto, read_features)
    return parsed_features["x"], parsed_features["y"]


def read_tf_records(abs_tf_record_path,
                    batch_size,
                    feature_sample_size,
                    label_sample_size,
                    dtype_features,
                    dtype_labels,
                    shuffle=False,
                    shuffle_buffer_size=10000,
                    skip_first_n=0):
    """
    reads for set of tf record files into a data set
    Args:
        abs_tf_record_path (str): absolute path to load the tf record file from
        batch_size (int):
        feature_sample_size (int):
        dtype_features (tf.DType): should match to what was wrote to tf records
        dtype_labels(tf.DType): should match to what was wrote to tf records
        shuffle (bool): whether to shuffle or not
        shuffle_buffer_size (int): how much items to shuffle
        skip_first_n (int): Optional num of records to skip at the beginning of the dataset
            default is 0
    Returns:
        tf.data.TFRecordDataset, an object representing our dataset.
    """

    dataset = tf.data.TFRecordDataset(abs_tf_record_path)
    dataset = dataset.map(
        lambda x: _parse_fn(x, feature_sample_size, label_sample_size, dtype_features, dtype_labels),
        num_parallel_calls=4
    )  # parse into tensors
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.skip(count=skip_first_n)
    dataset = dataset.prefetch(buffer_size=1)
    return dataset


def convert_tensor_to_dataset(tensor, batch_size):
    # convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(tensor)

    dataset = dataset.batch(batch_size)

    return dataset


def read_and_build_vocab(opened_file):
    """
    Create a mapping for each element of the vocab to its ID.
    Args:
        opened_file: Opened vocab file

    Returns:
        dict that maps elements to its ID
    """
    elements = opened_file.read().split()
    element_to_id = dict(zip(elements, range(len(elements))))
    return element_to_id
