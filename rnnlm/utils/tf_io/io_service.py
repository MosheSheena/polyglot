import tensorflow as tf

from rnnlm.utils.tf_io import reader, writer


def raw_to_tf_records(gen_data,
                      abs_tf_record_path,
                      preprocessor_feature_fn=None,
                      preprocessor_feature_params=None,
                      preprocessor_label_fn=None,
                      preprocessor_label_params=None):
    """
    convert raw data (sentences) into tf records format
    Args:
        gen_data (generator): function that defines how to iterate the data
        abs_tf_record_path (str): absolute path to tfrecord file of the data
        preprocessor_feature_fn (func): returns the preprocessed tensor
        preprocessor_feature_params (args*): additional args for preprocessor_feature_fn besides
         features, if there is any.
        preprocessor_label_fn (func): returns the preprocessed tensor
        preprocessor_label_params (args*): additional args for preprocessor_label_fn besides
         labels, if there is any.

    Returns:
        None
    """

    writer.write_tf_records(gen_words=gen_data,
                            destination_path=abs_tf_record_path,
                            preprocessor_feature_fn=preprocessor_feature_fn,
                            preprocessor_feature_params=preprocessor_feature_params,
                            preprocessor_label_fn=preprocessor_label_fn,
                            preprocessor_label_params=preprocessor_label_params)


def load_dataset(abs_tf_record_path, batch_size, seq_len, skip_first_n=0):
    """
    Load a tf record dataset using tf.data API
    Args:
        abs_tf_record_path (str): absolute path of the tf record file
        batch_size (int):
        seq_len (int):
        skip_first_n (int): how many records to skip from beginning of file

    Returns:
        tf.data.TFRecordDataset, an object representing our dataset.
    """
    return reader.read_tf_records(abs_tf_record_path=abs_tf_record_path,
                                  batch_size=batch_size,
                                  seq_len=seq_len,
                                  dtype_features=tf.int64,
                                  dtype_labels=tf.int64,
                                  skip_first_n=skip_first_n)
