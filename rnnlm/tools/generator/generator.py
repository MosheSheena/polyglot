#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module meant to enable word / sentence generation using a model
"""
import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

# For importing rnnlm module
sys.path.append(os.getcwd())
from rnnlm.tools.generator import helper, config

ADD_START_FLAG = True
LSTM_CONTEXT = None


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
        help="Absolute path to wordlist.rnn.id file",
        required=True
    )
    parser.add_argument(
        "-d", "--dest",
        help="Absolute path to generate words file",
        required=False
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        help="Number of words which afterwards the context will be reset. Higher number results in "
             "higher computation time",
        default=100,
        required=False
    )

    # Get arguments from command line
    args = parser.parse_args()
    model_path = args.model
    words_path = args.words
    gen_file_path = args.dest
    context_limit = args.limit

    return model_path, words_path, gen_file_path, context_limit


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


def get_language_model_probabilities(sess, word, tensors, context=None):
    """
    Given a word, return all probabilities to predict following words from the word vocabulary

    Args:
        sess (tf.Session): a session to use running the ops in this function
        word (tf.int32): a word id representing a word from the vocabulary
        tensors (dict): a dict mapping tensor names to tensors required for the generation
        context (tf.Tensor): a tensor of shape [?, ?] representing the state of the LSTM

    Returns:
        tf.Tensor: a tensor of shape [1, <vocabulary_length>] containing all probabilities for words
        to follow the given word

    """
    word_tensor = [[word]]  # encapsulating the word_id in shape (1, 1)

    if context is None:
        context = sess.run(tensors["initial_state"])

    fetches = {
        "state_out": tensors["state_out"],
        "cell_out": tensors["cell_out"]
    }

    feed_dict = {
        tensors["word_in"]: word_tensor,
        tensors["state_in"]: context
    }

    run_res_dict = sess.run(fetches, feed_dict)

    feed_dict[tensors["word_out"]] = word_tensor
    feed_dict[tensors["cell_in"]] = run_res_dict["cell_out"]

    logits = tf.matmul(
        a=tensors["cell_in"],
        b=tensors["softmax_w"]
    ) + tensors["softmax_b"]

    logits = sess.run(logits, feed_dict)
    
    # probabilities_tensor = sess.run(tf.nn.softmax(logits=logits), feed_dict)
    context = run_res_dict["state_out"]

    # return probabilities_tensor, context
    return logits, context


def generate_word(
    sess,
    word,
    word_2_id,
    id_2_word,
    tensors,
    context=None,
    temperature=None
):
    """
    Given a single word, generate a word using a language model

    Args:
        sess (tf.Session):
        word (str):
        word_2_id (dict):
        id_2_word (dict):
        tensors (dict):
        context (tf.Tensor):
        temperature (float):

    Returns:
        tuple:

    """
    word_id = word_2_id.get(word, word_2_id.get(config.OUT_OF_SCOPE))

    logits, context = get_language_model_probabilities(
        sess=sess,
        word=word_id,
        tensors=tensors,
        context=context
    )

    logits = np.array(logits)
    logits /= temperature if temperature else logits
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities)
    probabilities = probabilities.ravel()

    gen_word_index = np.random.choice(range(len(probabilities)), p=probabilities)
    gen_word = id_2_word.get(gen_word_index, config.OUT_OF_SCOPE)

    return gen_word, context


def choose_random_word(word_ids, probability_distribution=None):
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
    word = np.random.choice(a=word_ids, p=probability_distribution)

    # index = np.where(probability_tensor == probability)

    return word


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
    
    1. Generate interactively
    2. Generate a file
        """
    )
    option = input("Choose an option:")

    return int(option)


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


def collect_action_arguments(action):
    """
    Gather all required arguments for the chosen action

    Args:
        action (int): action number from the action menu
        
    Returns:
        dict: kwargs for action execution
    """
    def initial_word_settings():
        """
        Ask the user if he wants to provide an initial word or to choose one at random

        Returns:
            dict: a dictionary of the arguments required for executing an action
        """
        global LSTM_CONTEXT
        global ADD_START_FLAG

        arguments = {}

        _input = input("Enter a word/sentence: [</s>]")

        if _input.startswith("//"):
            LSTM_CONTEXT = None
            _input = _input[3:]
        if _input.startswith(r"\\"):
            ADD_START_FLAG = False
            _input = _input[3:]

        sentence = _input.split()
        if sentence:
            arguments["initial_input"] = sentence
        else:  # Using </s> as the initial word
            arguments["initial_input"] = [config.START]
            ADD_START_FLAG = False

        _temperature = input(
            "Enter temperature: [{def_temp}]".format(
                def_temp=config.DEFAULT_TEMPERATURE
            )
        )
        temperature = float(_temperature) if _temperature else config.DEFAULT_TEMPERATURE
        arguments["temperature"] = temperature

        return arguments

    args = initial_word_settings()

    if action == config.GEN_SENTENCE:
        sentence_len = input(
            "Enter number of words: [{def_len}]".format(
                def_len=config.DEFUALT_SENTENCE_LENGTH
            )
        )
        sentence_len = int(sentence_len) if sentence_len else config.DEFUALT_SENTENCE_LENGTH

        args["sentence_len"] = sentence_len

    elif action == config.GEN_SENTENCES:
        num_words = input(
            "Enter number of words: [{def_num}]".format(
                def_num=config.DEFUALT_NUM_WORDS_FOR_FILE
            )
        )
        num_words = num_words if num_words else config.DEFUALT_NUM_WORDS_FOR_FILE

        args["num_words"] = num_words

    return args


def execute_action(action_args, sess, word_2_id, id_2_word, tensors):
    """
    Execute an action using the language model.

    Args:
        action_args (dict):
        sess (tf.Session):
        word_2_id (dict):
        id_2_word (dict):
        tensors (dict):
    """
    global LSTM_CONTEXT
    global ADD_START_FLAG

    action = action_args["action"]
    words = action_args["initial_input"]

    words = convert_to_uppercase(words)
    words = check_insert_start_of_sentence_token(words)

    if action == config.GEN_SENTENCE:
        sentence_len = int(action_args["sentence_len"])

        last_word_in_sequence = feed_language_model(
            id_2_word=id_2_word,
            sess=sess,
            tensors=tensors,
            word_2_id=word_2_id,
            words=words,
            temperature=action_args["temperature"]
        )

        print_sentence = generate_words(
            id_2_word=id_2_word,
            sentence_len=sentence_len,
            sess=sess,
            tensors=tensors,
            word_2_id=word_2_id,
            initial_word=last_word_in_sequence,
            temperature=action_args["temperature"]
        )

        print(print_sentence)

    elif action == config.GEN_SENTENCES:
        num_words = int(action_args["num_words"])
        file_name = create_generation_file_name(action_args)

        last_word_in_sequence = feed_language_model(
            id_2_word=id_2_word,
            sess=sess,
            tensors=tensors,
            word_2_id=word_2_id,
            words=words,
            temperature=action_args["temperature"]
        )

        generate_sentences_file(
            action_args=action_args,
            file_name=file_name,
            id_2_word=id_2_word,
            num_words=num_words,
            sess=sess,
            tensors=tensors,
            word_2_id=word_2_id,
            initial_sequence=words,
            initial_word=last_word_in_sequence,
            temperature=action_args["temperature"]
        )


def generate_sentences_file(
    action_args,
    file_name,
    id_2_word,
    num_words,
    sess,
    tensors,
    word_2_id,
    initial_sequence,
    initial_word,
    temperature
):
    global LSTM_CONTEXT

    for word in initial_sequence:
        if word != config.START:
            write_to_file(file_name=file_name, word=word)
    last_word = initial_word

    for i in range(num_words):
        gen_word, LSTM_CONTEXT = generate_word(
            sess=sess,
            word=last_word,
            word_2_id=word_2_id,
            id_2_word=id_2_word,
            tensors=tensors,
            context=LSTM_CONTEXT,
            temperature=temperature
        )
        check_context_reset(limit=action_args["context_limit"], num_word=i)
        last_word = write_to_file(file_name=file_name, word=gen_word)


def write_to_file(file_name, word):
    next_char = '\n' if word == config.START else ' '

    with open(file=file_name, mode='a+') as file:
        file.write(word + next_char)
    return word


def check_context_reset(limit, num_word):
    global LSTM_CONTEXT

    if num_word % limit == 0:
        LSTM_CONTEXT = None


def create_generation_file_name(action_args):
    time_sig = create_time_signature()
    if action_args["gen_file_path"]:
        file_name = action_args["gen_file_path"] + time_sig + "_gen_words.txt"
    else:
        file_name = time_sig + "_gen_words.txt"
    return file_name


def create_time_signature():
    time_sig = "_".join(str(datetime.now())[:19].split())
    return time_sig


def generate_words(
    id_2_word,
    sentence_len,
    sess,
    tensors,
    word_2_id,
    initial_word,
    temperature
):
    """
    Generate a sequence of words using an initial word

    Args:
        id_2_word (dict): a dict mapping indices to words in the vocabulary
        sentence_len (int): required length to generated sentence
        sess (tf.Session): the current session
        tensors (dict): a dict mapping tensor names to tf.Tensor objects
        word_2_id (dict): a dict mapping words from the vocabulary to indices
        initial_word (str): initial input to the LM
        temperature (float): hyper parameter to divide the logits by

    Returns:
        str: the generated sentence
    """
    global LSTM_CONTEXT

    sequence = list()
    print_sentence = list()
    sequence.append(initial_word)
    print_sentence.append(initial_word)

    for i in range(sentence_len):
        gen_word, LSTM_CONTEXT = generate_word(
            sess=sess,
            word=sequence[i],
            word_2_id=word_2_id,
            id_2_word=id_2_word,
            tensors=tensors,
            context=LSTM_CONTEXT,
            temperature=temperature
        )
        print_sentence.append(gen_word)
    return ' '.join(print_sentence)


def feed_language_model(
    id_2_word,
    sess,
    tensors,
    word_2_id,
    words,
    temperature
):
    """
    Feed the LM with all words in the sequence except for the last one.
    The last word will be returned and used as the input for the LM to
    generate it's predictions

    Args:
        id_2_word (dict): a dict mapping indices to words in the vocabulary
        sess (tf.Session): the current session
        tensors (dict): a dict mapping tensor names to tf.Tensor objects
        word_2_id (dict): a dict mapping words from the vocabulary to indices
        words (list): a sequence of words
        temperature (float): hyper parameter to divide the logits by

    Returns:
        str: the last word in the sequence provided, so that will be passed as
        an input to the LM to make a prediction
    """
    global LSTM_CONTEXT

    if len(words) > 1:
        for word in words[:-1]:
            gen_word, LSTM_CONTEXT = generate_word(
                sess=sess,
                word=word,
                word_2_id=word_2_id,
                id_2_word=id_2_word,
                tensors=tensors,
                context=LSTM_CONTEXT,
                temperature=temperature
            )
    else:
        return words[0]
    return words[-1]


def strip_start_of_sentence(word_list):
    if word_list[0] is config.START:
        word_list.remove(config.START)
    return word_list

  
def check_insert_start_of_sentence_token(words):
    """
    Insert a </s> token to the start of a sequence of words if required

    Args:
        words (list): sequence of words

    Returns:
        list: sequence of words with or without </s>
    """
    global ADD_START_FLAG

    if ADD_START_FLAG:
        return [config.START] + words  # adding </s> for start of sentence
    else:
        ADD_START_FLAG = True
        return words


def convert_to_uppercase(words):
    words = list(map(lambda x: str(x).upper(), words))
    return words


def terminate():
    sys.exit(0)


def map_vocabulary_to_indices(word_path):
    """
    Maps a vocabulary of words to indices

    Args:
        word_path (str): absolute path to the vocabulary file. This file should
        be a word per line

    Returns:
        tuple: dicts mapping between words to indices and vice versa
    """
    word_2_id = parse_words_id_file_to_dict(file=word_path)
    id_2_word = create_reverse_dict(_dict=word_2_id)
    return id_2_word, word_2_id


def is_action_valid(selected_action):
    """
    Ensure the action chosen from the menu is valid

    Args:
        selected_action (int): the action selected

    Returns:
        bool: True if the action is valid. False otherwise
    """
    if selected_action < config.GEN_SENTENCE or selected_action > config.GEN_SENTENCES:
        return False
    return True


def prompt_termination():
    exit_generating = input("Quit interactive generation? Y/N [N]")
    if exit_generating is 'Y':
        terminate()


def main():
    model_path, word_path, gen_file_path, context_limit = get_args()

    graph, sess = load_trained_model(log_dir_path=model_path)
    tensors = helper.create_tensor_mapping(
        tensor_dict=config.TENSORS_OF_MODEL_DICT,
        graph=graph
    )

    id_2_word, word_2_id = map_vocabulary_to_indices(word_path=word_path)

    selected_action = None
    while selected_action != config.EXIT:
        selected_action = create_action_menu()
        if not is_action_valid(selected_action=selected_action):
            print("A non valid action was chosen")
            break

        args = collect_action_arguments(action=selected_action)
        args["action"] = selected_action

        if selected_action is config.GEN_SENTENCE:
            while True:
                execute_action(
                    action_args=args,
                    sess=sess,
                    word_2_id=word_2_id,
                    id_2_word=id_2_word,
                    tensors=tensors
                )
                prompt_termination()
                args = collect_action_arguments(action=config.GEN_SENTENCE)
                args["action"] = config.GEN_SENTENCE

        if selected_action is config.GEN_SENTENCES:
            args["gen_file_path"] = gen_file_path
            args["context_limit"] = context_limit
            execute_action(
                action_args=args,
                sess=sess,
                word_2_id=word_2_id,
                id_2_word=id_2_word,
                tensors=tensors
            )
            terminate()


if __name__ == "__main__":
    main()
