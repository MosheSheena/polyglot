import tensorflow as tf
from rnnlm.utils.tf_io import reader, io_service


def preprocess_elements_with_vocab(gen_fn,
                                   seq_len,
                                   abs_vocab_path_features,
                                   abs_vocab_path_labels,
                                   abs_raw_data_train,
                                   abs_raw_data_valid,
                                   abs_raw_data_test,
                                   abs_train_tf_record_path,
                                   abs_valid_tf_record_path,
                                   abs_test_tf_record_path):
    """
    Enumerates every single element in the data by its appearance in the vocab
    Args:
        gen_fn (func): function that yields the features and labels from raw data
        seq_len (int):
        abs_vocab_path_features (str): absolute path of the vocab for features
        abs_vocab_path_labels (str): absolute path of the vocab for labels
        abs_raw_data_train (str): absolute path of the original training data
        abs_raw_data_valid (str): absolute path of the original validation data
        abs_raw_data_test (str): absolute path of the original test data
        abs_train_tf_record_path (str): absolute path of output file for train
        abs_valid_tf_record_path (str): absolute path of output file for validation
        abs_test_tf_record_path (str): absolute path of output file for test

    Returns:
        None
    """
    vocab_features = build_vocab(abs_vocab_path_features)
    vocab_labels = build_vocab(abs_vocab_path_labels)
    with tf.gfile.GFile(abs_raw_data_train, 'r') as f:
        io_service.raw_to_tf_records(gen_raw_data=gen_fn(f, seq_len),
                                     abs_tf_record_path=abs_train_tf_record_path,
                                     preprocessor_feature_fn=map_elements_to_ids,
                                     preprocessor_feature_params=vocab_features,
                                     preprocessor_label_fn=map_elements_to_ids,
                                     preprocessor_label_params=vocab_labels)
    with tf.gfile.GFile(abs_raw_data_valid, 'r') as f:
        io_service.raw_to_tf_records(gen_raw_data=gen_fn(f, seq_len),
                                     abs_tf_record_path=abs_valid_tf_record_path,
                                     preprocessor_feature_fn=map_elements_to_ids,
                                     preprocessor_feature_params=vocab_features,
                                     preprocessor_label_fn=map_elements_to_ids,
                                     preprocessor_label_params=vocab_labels)
    with tf.gfile.GFile(abs_raw_data_test, 'r') as f:
        io_service.raw_to_tf_records(gen_raw_data=gen_fn(f, seq_len),
                                     abs_tf_record_path=abs_test_tf_record_path,
                                     preprocessor_feature_fn=map_elements_to_ids,
                                     preprocessor_feature_params=vocab_features,
                                     preprocessor_label_fn=map_elements_to_ids,
                                     preprocessor_label_params=vocab_labels)


def map_elements_to_ids(elements, vocab):
    """
    Convert elements to their id's in the vocab, vocab must contain
    an entry for elements that do not exist in the vocab with the key '<oos>'
    Args:
        elements (list): list of elements
        vocab (dict): a dict that maps each element to its ID

    Returns:
        list of IDs representing the original elements, any element that was not found
    """
    return [vocab[element] if element in vocab else vocab["<oos>"] for element in elements]


def build_vocab(vocab_file_path):
    with tf.gfile.GFile(vocab_file_path, 'r') as vocab_file:
        vocab = reader.read_and_build_vocab(vocab_file)
    return [vocab]
