import tensorflow as tf

from rnnlm.models.lstm_fast import reader
from rnnlm.models.lstm_fast import writer


def raw_to_tf_records(raw_path,
                      tf_record_path,
                      seq_len,
                      preprocessor_feature_fn=None,
                      preprocessor_feature_params=None,
                      preprocessor_label_fn=None,
                      preprocessor_label_params=None,
                      overlap=False):
    """
    convert raw data (sentences) into tf records format
    Args:
        raw_path (str): path to original file of data, before tf record conversion
        tf_record_path (str): path to tfrecord file of the data
        seq_len (int):
        overlap (bool): whether the data should be read with overlaps
        preprocessor_feature_fn (func): returns the preprocessed tensor
        preprocessor_feature_params (list): additional args for preprocessor_feature_fn besides
         features, if there is any.
        preprocessor_label_fn (func): returns the preprocessed tensor
        preprocessor_label_params (list): additional args for preprocessor_label_fn besides
         labels, if there is any.

    Returns:
        None
    """
    with tf.gfile.GFile(raw_path, 'r') as raw_file:
        if overlap:
            gen_words = reader.gen_shifted_words_with_overlap(file_obj=raw_file, seq_len=seq_len)
        else:
            gen_words = reader.gen_no_overlap_words(file_obj=raw_file, seq_len=seq_len)

        writer.write_tf_records(gen_words=gen_words,
                                destination_path=tf_record_path,
                                preprocessor_feature_fn=preprocessor_feature_fn,
                                preprocessor_feature_params=preprocessor_feature_params,
                                preprocessor_label_fn=preprocessor_label_fn,
                                preprocessor_label_params=preprocessor_label_params)


def load_dataset(tf_record_path, batch_size, seq_len, skip_first_n=0):
    return reader.read_tf_records(tf_record_path=tf_record_path,
                                  batch_size=batch_size,
                                  seq_len=seq_len,
                                  skip_first_n=skip_first_n)
