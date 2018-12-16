#!/usr/bin/env python3

"""
A module meant to enable word / sentence generation using a model
"""

# Imports
import argparse
import os
import random

import numpy as np
import tensorflow as tf

from rnnlm.tools.generator import (
    helper as gen_helper,
    config as gen_conf
)

# Constants
NUM_OPTIONS = 4

GEN_WORD = 1
GEN_SENTENCE = 2
GEN_SENTENCES = 3
EXIT = 4

ERROR = -1

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

def get_args():
    """
    This function parses and return arguments passed in from the cli

    Returns:
        tuple: a tuple of arguments parsed from STDIN

    """
    parser = argparse.ArgumentParser(
        description="\nA tool for generating words/sentences using a language model\n",
        add_help=True
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
        "-s", "--single",
        help="Generate a single word"
    )
    parser.add_argument(
        "-S", "--sentence",
        help="Generate a single sentence"
    )
    parser.add_argument(
        "-f", "--word-file",
        help="Generate a file of words"
    )
    parser.add_argument(
        "-F", "--sentence-file",
        help="Generate a file of sentences"
    )
    # Get arguments from command line
    args = parser.parse_args()
    model_path = args.model
    words_path = args.words

    if args.single is not None:


    return model_path, words_path


def find_file_by_extension(path, extension):
    """Find a file with a certain extension in a specific path

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


def get_language_model_probabilities(graph, sess, word, context=None):
    """
    Given a word, return all probabilities to predict following words from the word vocabulary

    Args:
        graph (tf.Graph): a graph of the trained language model
        sess (tf.Session): a session to use running the ops in this function
        word (tf.int32): a word id representing a word from the vocabulary
        context (tf.Tensor): a tensor of shape [?, ?] representing the state of the LSTM

    Returns:
        tf.Tensor: a tensor of shape [1, <vocabulary_length>] containing all probabilities for words
        to follow the given word

    """
    word_tensor = [[word]]  # encapsulating the word_id in shape (1, 1)

    tensor_mapping_dict = gen_helper.create_tensor_mapping(
        tensor_dict=TENSORS_OF_MODEL_DICT,
        graph=graph
    )

    if context is None:
        context = sess.run(tensor_mapping_dict["initial_state"])

    fetches = {
        "state_out": tensor_mapping_dict["state_out"],
        "cell_out": tensor_mapping_dict["cell_out"]
    }

    feed_dict = {
        tensor_mapping_dict["word_in"]: word_tensor,
        tensor_mapping_dict["state_in"]: context
    }

    run_res_dict = sess.run(fetches, feed_dict)

    feed_dict[tensor_mapping_dict["word_out"]] = word_tensor
    feed_dict[tensor_mapping_dict["cell_in"]] = run_res_dict["cell_out"]

    logits = tf.matmul(
        a=tensor_mapping_dict["cell_in"],
        b=tensor_mapping_dict["softmax_w"]
    ) + tensor_mapping_dict["softmax_b"]

    probabilities_tensor = sess.run(tf.nn.log_softmax(logits=logits), feed_dict)

    return probabilities_tensor, context


def choose_random_word(probability_tensor, probability_distribution=None):
    """
    Given a probability tensor shaped (1, <vocabulary_size>), choose a word using uniform random method

    Args:
        probability_tensor (Tensor): a tensor containing all probabilities for words in the vocabulary
        probability_distribution (numpy.ndarray): a numpy array in the size of probability_tensor.
                                                  used for a weighted choice.

    Returns:
        tuple: a tuple of the form (<probability>, <index>), where:
            * probability = the probability for the word chosen
            * index = the word's position within the tensor (which should correspond to the word's id)

    """
    if probability_distribution:
        probability = np.random.choice(a=probability_tensor, size=1, p=probability_distribution)
    else:
        probability = np.random.choice(a=probability_tensor, size=1)

    index = np.where(probability_tensor == probability)

    return probability, index


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


def create_reverse_dict(_dict):
    """
    Reverses the keys and values of the given dictionary

    Args:
        _dict (dict): a dictionary

    Returns:
        dict: a reversed dictionary

    """
    return dict(zip(_dict.values(), _dict.keys()))


def create_action_menu():
    """
    Print a menu with actions using the language model

    Returns:
        int: an option the user selected

    """
    print(
        """
    ******** Language-Model-Generator ********
    
    1. Generate a word
    2. Generate a sentence
    3. Generate a file of sentences
    4. Exit
        """
    )
    option = input("Choose an option by typing a number:")

    return option


def choose_a_random_word(id_2_word):
    """
    Choose a random word from the language model's vocabulary

    Args:
        id_2_word (dict): a dictionary mapping word id's to words

    Returns:
        str: a word from the vocabulary

    """
    word_count = len(id_2_word)
    random_id = random.randint(1, word_count)

    return id_2_word[random_id]


def collect_action_arguments(action, word_2_id, id_2_word):
    """
    Gather all required arguments for the chosen action

    Args:
        action (int): action number from the action menu
        word_2_id (dict): a dictionary mapping words to word id's
        id_2_word (dict): a dictionary mapping id's to vocabulary words
    Returns:
        dict: kwargs for action execution

    """
    args = {}
    if action == GEN_WORD:
        init_word_from_user = input("Provide initial word [Y/N]?")
        if init_word_from_user == 'y' or init_word_from_user == 'Y':
            word = input("Enter a word:")
            word = word_2_id.get(word, None)
            if word is None:
                print("word does not belong in the vocabulary")
                return ERROR
            else:
                args["word"] = word
                return args
        else:  # Choosing an initial word in random
            args["word"] = choose_a_random_word(id_2_word)

    elif action == GEN_SENTENCE:
        pass
    elif action == GEN_SENTENCES:
        pass





def execute_action(action):
    """

    Args:
        action:

    Returns:

    """



if __name__ == "__main__":
    # Parsing the cli args passed to the script
    model_path, word_path = get_args()  # parsing the cli flags

    # Loading the graph and session
    graph, sess = load_trained_model(log_dir_path=model_path)
    word_2_id_dict = parse_words_id_file_to_dict(file=word_path)
    id_2_word_dict = create_reverse_dict(_dict=word_2_id_dict)

    # Generating a choice menu and prompt the user for actions
    action = None
    while action != EXIT:
        action = create_action_menu()
        # Execute an action the user chose
        if not isinstance(action, int):
            print("Option invalid. please choose between 1 and {}".format(NUM_OPTIONS))
            continue  # return to loop condition


    #################################
    init_word = "THIS"
    init_word_id = word_2_id_dict[init_word]
    context = None
    num_words_to_generate = 5
    word_list = [init_word]

    for i in range(num_words_to_generate):
        prob_tensor, context = get_language_model_probabilities(
            graph=graph,
            sess=sess,
            word=word_2_id_dict[word_list[i]],
            context=context
        )
        # NOTE: prob_tensor[0] and gen_word_index[0][0] are for shape compatibility across functions
        gen_word_prob, gen_word_index = choose_random_word(probability_tensor=prob_tensor[0])

        word_list.append(id_2_word_dict[gen_word_index[0][0]])

    print("Generated sentence: {}".format(" ".join(word_list)))