from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import csv
from nltk import word_tokenize, pos_tag
from collections import defaultdict

READ_ENTIRE_FILE_MODE = -1


def _read_n_shifted_words_gen(file_obj, n, overlap=False):
    """
    Generator function that reads n words each time from file.
    Each yield contains a list of words shifted by 1
    if n = 0 it returns an empty list
    if n is negative it returns all the words from file_obj
    Args:
        file_obj: opened file
        n: (int) num of words to read from file

    Returns:
        list of n words from file
    """
    if n < 0:
        yield list(file_obj.read().split())
    elif n == 0:
        yield list()
    else:
        n_words = list()

        for line in file_obj:
            for word in line.split():
                n_words.append(word)
                if len(n_words) == n:
                    yield list(n_words)

                    if overlap:
                        # remove the first element
                        # from here and on one element will be inserted to the list and will be yield
                        n_words.pop(0)
                    else:
                        # flush all elements and get n new ones
                        n_words.clear()

        # take care of the remainder of num_words % n
        if len(n_words) % n != 0:
            yield n_words


def gen_shifted_words_with_overlap(file_obj, seq_len):

    gen_words = _read_n_shifted_words_gen(file_obj=file_obj, n=seq_len, overlap=True)
    x = next(gen_words)
    y = next(gen_words)
    while True:
        try:
            if x[1:] != y[:-1] or len(x) != len(y):
                raise StopIteration  # ignore remainder that is less than the sequence length

            yield (x, y)
            # x = y since the words are shifted by 1 in time
            x = y
            y = next(gen_words)

        except StopIteration:
            break


def gen_no_overlap_words(file_obj, seq_len):

    gen_words = _read_n_shifted_words_gen(file_obj=file_obj, n=seq_len, overlap=False)

    accumulator = next(gen_words)
    while True:
        try:
            x = accumulator
            accumulator = next(gen_words)
            y = x[1:] + [accumulator[0]]

            if x[1:] != y[:-1] or len(x) != len(y):
                raise StopIteration  # ignore remainder that is less than the sequence length

            yield (x, y)

        except StopIteration:
            break


def gen_pos_tagger(file_obj, seq_len, overlap=False):
    """
    Each call to this generator generates a tuple of x, y
    where x is list of words with a list size of seq_len
    and y is a list of part-of-speech tags of the words
    in x accordingly.
    Args:
        file_obj: opened file of raw data
        seq_len: the length of how much we will read from the
            file in each generation
        overlap: whether to generate with overlaps or not

    Returns:

    """

    gen_words = _read_n_shifted_words_gen(file_obj=file_obj, n=seq_len, overlap=overlap)
    num_of_iterations = 0
    count_diff_tags = defaultdict(int)
    csv_file = open("pos.csv", 'a+')
    csv_file.write("x,y\n")
    for words_list in gen_words:
        words_without_tags = list()

        # deal with special tags such as <oos>, <unk> and </s>
        # chars like < and > messes with the tagging therefore they are removed
        for w in words_list:
            if w.startswith('<') and w.endswith('>'):
                w = w[1:-1]
            words_without_tags.append(w)

        # the features are the words
        x = words_without_tags
        if len(x) < seq_len:
            print("num of diff tags = {}".format(len(count_diff_tags)))
            print("tags = {}".format(count_diff_tags))
            csv_file.close()
            raise StopIteration
        # the labels are the pos tags of those words
        y = list()

        words_str = " ".join(words_without_tags)
        tokens = word_tokenize(words_str)
        pos_tagged_words = pos_tag(tokens)

        if len(pos_tagged_words) > len(words_without_tags):  # then we have some words that got more than one tag

            it_pos = iter(pos_tagged_words)
            for w in words_without_tags:
                if len(word_tokenize(w)) > 1 and len(w) > 2:

                    next_pos_tagged = next(it_pos)

                    before_apostrophe_tag = next_pos_tagged[1]
                    next_pos_tagged = next(it_pos)
                    after_apostrophe_tag = next_pos_tagged[1]

                    # concat tags to create a single cat that will be used for classification
                    tag = before_apostrophe_tag + '_' + after_apostrophe_tag

                else:
                    tag = next(it_pos)[1]

                csv_file.write("{},{}\n".format(w, tag))
                y.append(tag)
                count_diff_tags[tag] += 1

        else:  # len(tokens) <= len(words_without_tags)
            for w, tag in pos_tagged_words:
                csv_file.write("{},{}\n".format(w, tag))
                y.append(tag)
                count_diff_tags[tag] += 1

        assert len(y) == 20, "y of size{} = {}".format(len(y), y)

        num_of_iterations += 1
        yield x, y


def _parse_fn(example_proto, seq_len):
    """
    Parses a single example from tf record files into Tensor or SparseTensor

    Args:
        example_proto: (tf.train.Example) example from tf record file

    Returns:
        if tf.FixedLenFeature -> return Tensor
        if tf.VarLenFeature -> return SparseTensor
    """
    read_features = {
        "x": tf.FixedLenFeature(shape=[seq_len], dtype=tf.int64),
        "y": tf.FixedLenFeature(shape=[seq_len], dtype=tf.int64),
    }
    parsed_features = tf.parse_single_example(example_proto, read_features)
    return parsed_features["x"], parsed_features["y"]


def read_tf_records(tf_record_path, batch_size, seq_len, shuffle=False):
    """
    reads tf record file into a data set
    Args:
        tf_record_path (str): where to load the tf record file from
        batch_size (int):
        seq_len (int):
        shuffle (bool):
    Returns:
        next_op for one shot iterator of the dataset
    """

    dataset = tf.data.TFRecordDataset(tf_record_path)
    dataset = dataset.map(lambda x: _parse_fn(x, seq_len), num_parallel_calls=4)  # parse into tensors
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.prefetch(buffer_size=1)
    return dataset.make_one_shot_iterator().get_next()


def build_vocab(file_obj):
    gen_words = _read_n_shifted_words_gen(file_obj, READ_ENTIRE_FILE_MODE)
    words = next(gen_words)
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id
