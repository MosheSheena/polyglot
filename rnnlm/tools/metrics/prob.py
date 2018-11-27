#!/usr/bin/env python3

"""
A module meant to enable measuring the probabilities of words / sentences using a language model
"""

# Imports
import argparse
import collections
import logging
import os
import sys
from functools import reduce
from math import exp

import tensorflow as tf

# Tensors required to load from model
TENSORS_OF_MODEL_DICT = {
    "word_in": "Test/Model/lstm_fast/test_word_in",
    "word_out": "Test/Model/lstm_fast/test_word_out",
    "initial_state": "Test/Model/lstm_fast/test_initial_state",
    "state_in": "Test/Model/lstm_fast/test_state_in",
    "state_out": "Test/Model/lstm_fast/test_state_out",
    "cell_in": "Test/Model/lstm_fast/test_cell_in",
    "cell_out": "Test/Model/lstm_fast/test_cell_out",
    "test_out": "Test/Model/lstm_fast/test_out",
    "softmax_w": "Model/lstm_fast/softmax_w",
    "softmax_b": "Model/lstm_fast/softmax_b"
}

# options in the menu
INTERACTIVE_MODE = 1
TEST_FILE_MODE = 2
EXIT = 3

UNK = '<unk>'

# global context to be adjusted as the program is running
LM_CONTEXT = None


def get_args():
    """This function parses and return arguments passed in the cli

    Returns:
        tuple: a tuple of

    """
    parser = argparse.ArgumentParser(
        description="\nA script that computes the probability for a word/sentence\n"
    )
    # Add arguments
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Absolute path to the model directory containing checkpoint and meta files",
        required=True
    )
    parser.add_argument(
        "-w", "--words",
        help="Absolute path to wordlist.rnn.final file",
        required=True
    )
    parser.add_argument(
        "-d", "--data",
        type=str,
        help="Absolute path to file containing words/sentences (ONE PER LINE)"
    )
    # Get arguments from command line
    args = parser.parse_args()
    model_path = args.model
    words_path = args.words
    data_path = args.data

    return model_path, words_path, data_path


def find_file_by_extension(path, extension):
    """Find a file with a certain extension in a specific path

    Args:
        path (str): path to look in
        extension (str): extension type

    Returns:
        str: the file's path if exists, otherwise "" (empty string)

    """
    logger = logging.getLogger(__name__)
    logger.info("Looking for a file with .{} extension in path: {}".format(extension, path))

    for file in os.listdir(path=path):
        if file.endswith(extension):
            logger.info("Found matching file: {}".format(file))
            return os.path.join(path, file)

    logger.error("No file with .{} extension in path: {}".format(extension, path))
    return ""


def load_trained_model(log_dir_path):
    """Load a model from the provided model_path

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
    logger = logging.getLogger(__name__)
    logger.info("Loading model from path: {}".format(log_dir_path))

    sess = tf.Session()
    meta_file_path = find_file_by_extension(path=log_dir_path, extension='meta')
    if not meta_file_path:
        logger.error("No .meta file in provided path")
        raise FileNotFoundError("No .meta file in provided path")

    saver = tf.train.import_meta_graph(meta_graph_or_file=meta_file_path)
    saver.restore(
        sess=sess,
        save_path=tf.train.latest_checkpoint(checkpoint_dir=log_dir_path)
    )
    graph = tf.get_default_graph()

    if graph:
        logger.info("Graph and Session successfully restored")
        return graph, sess
    else:
        return None


def create_tensor_mapping(tensor_dict, graph):
    """Create a dict mapping Tensor names (as saved from graph - including scope) to Tensor objects

    Args:
        tensor_dict (dict): dict containing mapping between tensor names and their name under scope,
        e.g: "word_in": "Test/Model/lstm_fast/test_word_in",
        graph (tf.Graph): a graph of the trained language model

    Returns:
        dict:
        dict containing mapping between Tensor names and their Tensor objects obtained from the graph

    """
    return {
        tensor_name: graph.get_tensor_by_name(tensor_dict[tensor_name] + ":0")
        for tensor_name in tensor_dict.keys()
    }


def parse_words_id_file_to_dict(file):
    """
    Transforms a file of the form:
    <ID> <word> per each line, to a set of id's mapping to words in a vocabulary

    Args:
        file (str): Absolute path to word-id file

    Returns:
        dict: a dictionary mapping words to id's

    """
    words_dict = {}

    with open(file=file, mode='r') as f:
        for line in f:
            line_split = line.strip().split()  # for removing '/n' and space
            word = line_split[1]
            word_id = int(line_split[0])
            words_dict[word] = word_id

    return words_dict


def create_reverse_dict(id_to_word_dict):
    """
    Reverses the keys and values of the given dictionary

    Args:
        id_to_word_dict (dict): a dictionary

    Returns:
        dict: a reversed dictionary

    """
    return dict(zip(id_to_word_dict.values(), id_to_word_dict.keys()))


def _file_to_list(file):
    """Convert file of words to a list of words

    Args:
        file (file): file to scan
            NOTE: the file should be in format of one word per line.

    Returns:
        list: list of words

    """
    word_list = []
    with open(file, 'r') as file:
        for line in file:
            tmp_line = line.split(' ')
            for word in tmp_line:
                word_list.append(word)
    return word_list


def _words_to_ids(words, id_dict):
    """Turn a list of words to a list of id's according to the id_dict param

    Args:
        words (list): list of words
        id_dict (dict): dict mapping words to id's

    Returns:
        list: a list of same length as words param where all elements are id's (int) corresponding to their words

    """
    return [id_dict.get(word, id_dict.get(UNK)) for word in words]


def build_id_dicts_for_prob(words_file):
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
    word_list = list(map(lambda x: x.strip(), word_list_raw))
    count = collections.Counter(word_list).most_common()
    dictionary = {}
    reverse_dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


def build_prob_dataset(data_file):
    """Build a matrix of words / sentences to check their probabilities using the language model

    Args:
        data_file (file): file containing the data for the probability computation.
                          each line contains a sentence or a single word

    Returns:
        list: a matrix where each row is a list of words that represent a sentence

    """
    sentence_matrix = []

    with open(data_file, 'r') as f:
        for line in f:
            words_list = line.split()
            sentence = list(map(lambda word: word.strip(), words_list))
            sentence_matrix.append(sentence)

    return sentence_matrix


def prepare_test_batch(sentence, id_dict):
    """Given a sentence prepare a list of tuples where each tuple is of the form (word_in, word_out)

    Args:
        sentence (list): list of words that represent a sentence
        id_dict (dict): a dict mapping words to ids

    Returns:
        list: a list of tuples

    """
    batch_list = []
    word_in_id = 0  # 0 is the id for '</s>' (start-of-sentence)

    id_list = _words_to_ids(words=sentence, id_dict=id_dict)

    for _id in id_list:
        batch_list.append((word_in_id, _id))
        word_in_id = _id

    return batch_list


def get_log_prob(graph, sess, word_in, word_out, context=None):
    """Given the parameters, calculates the probability of (word_out | word_in)

    Args:
        graph (tf.Graph): the graph object representing the DNN (Deep Neural Network)
        sess (tf.Session): a session to use running the ops in this function
        word_in (tf.int32): an id representing a word from the vocabulary
        word_out (tf.int32): an id representing a word from the vocabulary
        context (tf.Tensor): a tensor of shape [?, ?] representing the state of the LSTM

    Returns:
        Tuple: a tuple of (probability, context) where:
            * probability is the probability of (word_out | word_in)
            * context is the final state of the LSTM after computing probability

    """
    word_in_tensor = [[word_in]]    # to give shape of [1, 1]
    word_out_tensor = [[word_out]]  # to give shape of [1, 1]

    tensor_mapping = create_tensor_mapping(
        tensor_dict=TENSORS_OF_MODEL_DICT,
        graph=graph
    )

    if context is None:
        context = sess.run(tensor_mapping["initial_state"])

    fetches = {
        "state_out": tensor_mapping["state_out"],
        "cell_out": tensor_mapping["cell_out"]
    }

    feed_dict = {
        tensor_mapping["word_in"]: word_in_tensor,
        tensor_mapping["state_in"]: context
    }

    run_res_dict = sess.run(fetches, feed_dict)

    feed_dict[tensor_mapping["word_out"]] = word_out_tensor
    feed_dict[tensor_mapping["cell_in"]] = run_res_dict["cell_out"]

    probability = sess.run(tensor_mapping["test_out"], feed_dict)
    context = run_res_dict["state_out"]

    return probability, context


def main():
    # globals
    global LM_CONTEXT

    # ========== LOGGING SETUP ==========
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler('prob.log')
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    # ========== LOGGING SETUP ==========

    model_path, word_path, data_path = get_args()  # parsing the cli flags

    logger.info("Loading the model...")
    graph, sess = load_trained_model(log_dir_path=model_path)

    _dict = parse_words_id_file_to_dict(file=word_path)
    reversed_dict = create_reverse_dict(id_to_word_dict=_dict)

    action = create_action_menu()
    if action == INTERACTIVE_MODE:

        continue_flag = True

        while continue_flag:
            sentence_string = input("Enter a word/sentence: ")
            sentence = sentence_string.split()
            num_words = len(sentence)

            batch = prepare_test_batch(sentence=sentence, id_dict=_dict)
            log_probs_of_sentence = []

            for word_in_id, word_out_id in batch:
                prob, LM_CONTEXT = get_log_prob(
                    graph=graph,
                    sess=sess,
                    word_in=word_in_id,
                    word_out=word_out_id,
                    context=LM_CONTEXT
                )
                log_probs_of_sentence.append(prob)

            sentence_log_prob = reduce(lambda x, y: x+y, log_probs_of_sentence)
            average_sentence_log_prob = sentence_log_prob / num_words

            print("""
                =========================
                Sentence -> {sentence}
                Log probabilty -> {log_prob}
                Average log probabilty -> {avg_log_prob}
                =========================
                """.format(
                    sentence=" ".join(sentence),
                    log_prob=sentence_log_prob,
                    avg_log_prob=average_sentence_log_prob
                )
            )

            continue_input = input("continue Y/N? [Y]")
            if continue_input == 'N' or continue_input == 'n':
                continue_flag = False

    elif action == TEST_FILE_MODE:

        data_matrix = build_prob_dataset(data_path)

        logger.info("------------------------------------------------------")

        for sentence in data_matrix:
            logger.info("Computing probability for sentence: {}".format(" ".join(sentence)))
            batch = prepare_test_batch(sentence=sentence, id_dict=_dict)

            log_probs_of_sentence = []

            for word_in_id, word_out_id in batch:
                prob, LM_CONTEXT = get_log_prob(
                    graph=graph,
                    sess=sess,
                    word_in=word_in_id,
                    word_out=word_out_id,
                    context=LM_CONTEXT
                )

                logger.info(
                    "prob({word_out} | {word_in})={prob}".format(
                        word_out=reversed_dict[word_out_id],
                        word_in=reversed_dict[word_in_id],
                        prob=prob
                    )
                )

                log_probs_of_sentence.append(prob)

            log_prob_for_entire_sentence = reduce(lambda x, y: x + y, log_probs_of_sentence)
            exp_log_prob_for_entire_sentence = exp(log_prob_for_entire_sentence)
            logger.info("log prob for sentence: {}={}".format(" ".join(sentence), log_prob_for_entire_sentence))
            logger.info(
                "exp of log prob for sentence: {}={}".format(" ".join(sentence), exp_log_prob_for_entire_sentence))
            logger.info("------------------------------------------------------")

            sys.exit(0)

    elif action == EXIT:
        sys.exit(0)
    else:
        sys.exit(1)


def create_action_menu():
    """
    Print a menu with actions using the language model

    Returns:
        int: an option the user selected

    """
    print(
        """
    ******** Language-Model-Probabilities-Estimator ********

    1. Work interactive
    2. Work using a test file
    3. Exit
        """
    )
    option = input("Choose an option:")

    return int(option)


if __name__ == "__main__":
    main()
