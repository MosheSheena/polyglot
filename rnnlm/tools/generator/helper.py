"""
A helper file containing all functions to aid in word generation process
"""
import collections
import os

import tensorflow as tf


def find_file_by_extension(path, extension):
    """
    Find a file with a certain extension in a specific path

    Args:
        path (str): path to look in
        extension (str): extension type

    Returns:
        str: the file's path if exists, otherwise "" (empty string)

    """
    for file in os.listdir(path=path):
        if file.endswith(extension):
            return os.path.join(path, file)
    return ""


def load_trained_model(log_dir_path):
    """
    Load a model from the provided model_path

    Args:
        log_dir_path (str): A path where the model was saved, using a Saver
        object the path should contain both a .meta file and a checkpoint file.
        If unprovided the function will look in the current working directory.

    Returns:
        tuple: (<graph>, <session>) where graph is the loaded graph, and
        session is a tf.session object containing the graph.
        In the case where the model_path provided does not contain a valid
        model - None will be returned.

    """
    sess = tf.Session()
    meta_file_path = find_file_by_extension(path=log_dir_path, extension='meta')
    if not meta_file_path:
        raise FileNotFoundError("No .meta file in provided path")

    saver = tf.train.import_meta_graph(meta_graph_or_file=meta_file_path)
    saver.restore(
        sess=sess,
        save_path=tf.train.latest_checkpoint(checkpoint_dir=log_dir_path)
    )
    graph = tf.get_default_graph()

    if graph:
        return graph, sess
    else:
        return None


def _file_to_list(file):
    """
     Convert file of words to a list of words

    Args:
        file (file): file to scan
            NOTE: the file should be in format of one word per line.

    Returns:
        list: list of words

    """
    word_list = []
    with open(file, 'r') as file:
        for word in file:
            word_list.append(word)
    return word_list


def build_dataset_for_generation(words_file):
    """
    Builds a dataset from the list of words suitable for generating
    new words using a language model

    Args:
        words_file (file): a file containing the vocabulary of words that were
        used to train the model. one word per line

    Returns:
        tuple: (dict1, dict2) where dict1 is of the form <word, index>,
        and dict2 is of the form <index, word>

    """
    word_list_raw = _file_to_list(words_file)
    word_list = list(map(lambda s: s.strip(), word_list_raw))
    count = collections.Counter(word_list).most_common()
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


def create_tensor_mapping(tensor_dict, graph):
    """
    Create a dict mapping Tensor names (as saved from graph - including scope) to Tensor objects

    Args:
        tensor_dict (dict): dict containing mapping between tensor names and their name under scope,
        e.g: "word_in": "Test/Model/lstm_fast/test_word_in",
        graph (tf.Graph): a graph of the trained language model

    Returns:
        dict: dict containing mapping between Tensor names and their Tensor objects obtained from the graph

    """
    return {
        tensor_name: graph.get_tensor_by_name(tensor_dict[tensor_name] + ":0")
        for tensor_name in tensor_dict.keys()
    }
