import tensorflow as tf
from rnnlm.utils.tf_io import reader


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
