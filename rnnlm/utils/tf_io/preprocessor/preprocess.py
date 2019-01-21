import tensorflow as tf
from rnnlm.utils.tf_io import reader, io_service


def preprocess_elements_with_vocab(gen_fn,
                                   abs_vocab_path_features,
                                   abs_vocab_path_labels,
                                   abs_output_tf_record_path):
    """
    Enumerates every single element in the data by its appearance in the vocab
    Args:
        gen_fn (func): function that yields the features and labels from raw data
        abs_vocab_path_features (str): absolute path of the vocab for features
        abs_vocab_path_labels (str): absolute path of the vocab for labels
        abs_output_tf_record_path (str): absolute path of output file

    Returns:
        None
    """
    vocab_features = build_vocab(abs_vocab_path_features)
    vocab_labels = build_vocab(abs_vocab_path_labels)
    io_service.raw_to_tf_records(gen_fn_raw_data=gen_fn,
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
        vocab = reader.read_and_build_vocab(vocab_file)
    return vocab
