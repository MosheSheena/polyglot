"""
A helper file containing all functions to aid in word generation process

"""
# import statements
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
            return os.path.join(os.getcwd(), file)
    return ""


def load_trained_model(model_path=None):
    """
    Load a model from the provided model_path

    Args:
        model_path (str): A path where the model was saved.
        The path should contain both a .meta file and a checkpoint file.
        If not provided the function will look in the current working directory.

    Returns:
        tuple: (<graph>, <session>) where graph is the loaded graph, and
        session is a tf.session object containing the graph.
        In the case where the model_path provided does not contain a valid
        model - None will be returned.

    """
    sess = tf.Session()
    path = model_path if model_path else os.getcwd()
    saver = tf.train.import_meta_graph(meta_graph_or_file=path)
    saver.restore(
        sess=sess,
        save_path=tf.train.latest_checkpoint(path)
    )
    graph = tf.get_default_graph()

    if graph:
        return graph, sess
    else:
        return None
