import tensorflow as tf

from rnnlm.utils.tf_io import reader, writer


def raw_to_tf_records(gen_fn_raw_data,
                      abs_tf_record_path,
                      preprocessor_feature_fn=None,
                      preprocessor_label_fn=None):
    """
    convert raw data (sentences) into tf records format
    Args:
        gen_fn_raw_data (generator): function that defines how to iterate the data
        abs_tf_record_path (str): absolute path to tfrecord file of the data
        preprocessor_feature_fn (func): returns the preprocessed tensor
        preprocessor_label_fn (func): returns the preprocessed tensor

    Returns:
        None
    """

    writer.write_tf_records(gen_fn_words=gen_fn_raw_data,
                            destination_path=abs_tf_record_path,
                            preprocessor_feature_fn=preprocessor_feature_fn,
                            preprocessor_label_fn=preprocessor_label_fn)


def load_dataset(abs_tf_record_path,
                 batch_size,
                 seq_len,
                 dtype_features=tf.int64,
                 dtype_labels=tf.int64,
                 shuffle=False,
                 shuffle_buffer_size=10000,
                 skip_first_n=0):
    """
    Load a tf record dataset using tf.data API
    Args:
        abs_tf_record_path (str): absolute path of the tf record file
        batch_size (int):
        seq_len (int):
        dtype_features (tf.DType): should match to what was wrote to tf records
        dtype_labels(tf.DType): should match to what was wrote to tf records
        shuffle (bool): whether to shuffle the data
        shuffle_buffer_size (int): how much items to shuffle each time
        skip_first_n (int): how many records to skip from beginning of file

    Returns:
        tf.data.TFRecordDataset, an object representing our dataset.
    """
    return reader.read_tf_records(abs_tf_record_path=abs_tf_record_path,
                                  batch_size=batch_size,
                                  seq_len=seq_len,
                                  dtype_features=dtype_features,
                                  dtype_labels=dtype_labels,
                                  shuffle=shuffle,
                                  shuffle_buffer_size=shuffle_buffer_size,
                                  skip_first_n=skip_first_n)


def create_dataset_from_tensor(tensor, batch_size):
    """
    create a tf.data.Dataset from tensors
    Args:
        tensor: where each key is a name of the tensor and the value is
        the tensor itself
        batch_size (int): the batch size

    Returns:
        tf.data.Dataset
    """
    return reader.convert_tensor_to_dataset(tensor=tensor,
                                            batch_size=batch_size)


def create_vocab(file_obj):
    return reader.read_and_build_vocab(file_obj)

