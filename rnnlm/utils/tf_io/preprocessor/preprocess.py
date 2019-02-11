import tensorflow as tf
from rnnlm.utils.tf_io import io_service


def preprocess_with_function(gen_data_fn,
                             preprocess_features_fn,
                             preprocess_labels_fn,
                             abs_tf_record_output_path):
    """
    preprocess the data with a desired functions. The data is preprocessed and writen to the
    abs_tf_record_output_path
    Args:
        gen_data_fn(generator): generator that yields the data
        preprocess_features_fn(func): callable function that will preprocess the features
         the function needs to receive a chunk of features and preprocess it
        preprocess_labels_fn(func): callable function that will preprocess the labels
         the function needs to receive a chunk of labels and preprocess it
        abs_tf_record_output_path(str): absolute path for the output of the tf record file

    Returns:
        None
    """
    io_service.raw_to_tf_records(extractor_raw_data=gen_data_fn,
                                 abs_tf_record_path=abs_tf_record_output_path,
                                 preprocessor_feature_fn=preprocess_features_fn,
                                 preprocessor_label_fn=preprocess_labels_fn)


def preprocess_elements_with_vocab(extractor,
                                   abs_vocab_path_features,
                                   abs_vocab_path_labels,
                                   abs_output_tf_record_path):
    """
    Enumerates every single element in the data by its appearance in the vocab
    Args:
        extractor (generator): generator that yields the features and labels from raw data
        abs_vocab_path_features (str): absolute path of the vocab for features
        abs_vocab_path_labels (str): absolute path of the vocab for labels
        abs_output_tf_record_path (str): absolute path of output file

    Returns:
        None
    """
    vocab_features = build_vocab(abs_vocab_path_features)
    vocab_labels = build_vocab(abs_vocab_path_labels)
    io_service.raw_to_tf_records(extractor_raw_data=extractor,
                                 abs_tf_record_path=abs_output_tf_record_path,
                                 preprocessor_feature_fn=lambda x: map_elements_to_ids(x, vocab_features),
                                 preprocessor_label_fn=lambda y: map_elements_to_ids(y, vocab_labels))


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
        vocab = io_service.create_vocab(vocab_file)
    return vocab
