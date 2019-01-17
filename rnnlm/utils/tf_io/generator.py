from rnnlm.utils.pos import gen_pos_dataset

READ_ENTIRE_FILE_MODE = -1


def _gen_read_n_shifted_elements(file_obj, n, overlap=False):
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
        # if len(n_words) % n != 0:
        #     yield n_words


def gen_shifted_words_with_overlap(file_obj, seq_len):

    gen_words = _gen_read_n_shifted_elements(file_obj=file_obj, n=seq_len, overlap=True)
    x = next(gen_words)
    y = next(gen_words)
    while True:
        try:
            yield (x, y)
            # x = y since the words are shifted by 1 in time
            x = y
            y = next(gen_words)

        except StopIteration:
            break


def gen_no_overlap_words(file_obj, seq_len):

    gen_words = _gen_read_n_shifted_elements(file_obj=file_obj, n=seq_len, overlap=False)

    accumulator = next(gen_words)
    while True:
        try:
            x = accumulator
            accumulator = next(gen_words)
            y = x[1:] + [accumulator[0]]
            yield (x, y)

        except StopIteration:
            break


def gen_pos_tagger(file_obj, seq_len):
    """
    Each call to this generator generates a tuple of x, y
    where x is list of words with a list size of seq_len
    and y is a list of part-of-speech tags of the words
    in x accordingly.
    Args:
        file_obj: opened file of raw data, we  a
        seq_len: the length of how much we will read from the
            file in each generation

    Returns:

    """

    gen_words = _gen_read_n_shifted_elements(file_obj=file_obj, n=seq_len)
    words = list()
    tags = list()
    for word, tag in gen_pos_dataset(gen_words):

        words.append(word)
        tags.append(tag)
        if len(words) == seq_len:
            yield words, tags
            words.clear()
            tags.clear()
