from rnnlm.utils.pos import gen_pos_dataset


def _gen_next_word(file_obj):
    """
    Yields all the words from an opened file, one by one, without loading line by line
    Args:
        file_obj: an opened file

    Yields:
        the next word from the file
    """
    last = ""
    while True:
        buf = file_obj.read(3000)
        if not buf:
            break
        words = (last+buf).split()
        last = words.pop()
        for word in words:
            yield word
    if last:
        yield last


def _gen_n_words(file_obj,
                 n,
                 shift=False,
                 num_shifts=0):
    """
    Generates n words from file. if shift flag is true than num_shifts arg is
    expected, the function will yield words shifted by num_shifts.
    E.g given the raw text data:
    "WONDER HOW MUCH OF THE MEETINGS IS TALKING ABOUT THE STUFF AT THE MEETINGS </s>
    YEAH </s>"
    and the following args passed: n=4 , shift=False
    the function will yield (ignoring remainder):
     [
         ['WONDER', 'HOW', 'MUCH', 'OF'],
         ['THE', 'MEETINGS', 'IS', 'TALKING'],
         ['ABOUT', 'THE', 'STUFF', 'AT'],
         ['THE', 'MEETINGS', '</s>', 'YEAH']
     ]

    Given the same raw data but this time n=4 , shift=True, num_shifts=1
    [
         ['WONDER', 'HOW', 'MUCH', 'OF'],
         ['HOW', 'MUCH', 'OF', 'THE'],
         ['MUCH', 'OF', 'THE', 'MEETINGS'],
         ['OF', 'THE', 'MEETINGS', 'IS']
         ['THE', 'MEETINGS', 'IS', 'TALKING'],
         ['MEETINGS', 'IS', 'TALKING', 'ABOUT'],
         ['IS', 'TALKING', 'ABOUT', 'THE'],
         ['TALKING', 'ABOUT', 'THE', 'STUFF'],
         ['ABOUT', 'THE', 'STUFF', 'AT'],
         ['THE', 'STUFF', 'AT', 'THE'],
         ['STUFF', 'AT', 'THE', 'MEETINGS'],
         ['AT', 'THE', 'MEETINGS', '</s>'],
         ['THE', 'MEETINGS', '</s>', 'YEAH'],
         ['MEETINGS', '</s>', 'YEAH', '</s>']
     ]
    Args:
        file_obj (file): opened file
        n (int): num of words to read from file
        shift (bool): whether to yield shifted words from file instead on yielding
         n different each time
        num_shifts (int): how many elements to shift between each yield

    Yields:
        list of n words from file
    """

    n_words = list()

    gen_read_words = _gen_next_word(file_obj)
    for word in gen_read_words:
        n_words.append(word)
        if len(n_words) == n:
            yield list(n_words)

            if shift:
                # pop elements until shift condition is met
                for i in range(num_shifts):
                    n_words.pop(0)
            else:
                # flush all elements and get n new ones
                n_words.clear()

    # take care of the remainder of num_words % n
    # if len(n_words) % n != 0:
    #     yield n_words


def extract_without_overlap_in_same_and_between_yields(file_obj, seq_len):
    """
    Generates x, y where each element is shifted by seq_len
    E.g given the raw text data:
    "WONDER HOW MUCH OF THE MEETINGS IS TALKING ABOUT THE STUFF AT THE MEETINGS </s>
    YEAH </s>"
    and seq_len=4
    the function will yield (ignoring remainder):

    Yield #1:
      x = ['WONDER', 'HOW', 'MUCH', 'OF'], y = ['THE', 'MEETINGS', 'IS', 'TALKING']
    Yield #2:
      x = ['ABOUT', 'THE', 'STUFF', 'AT'], y = ['THE', 'MEETINGS' '</s>', 'YEAH']

    Args:
        file_obj (file): opened file
        seq_len (int):

    Yields:

    """
    gen_w = _gen_n_words(file_obj=file_obj, n=seq_len)
    while True:
        try:
            x = next(gen_w)
            y = next(gen_w)
            yield list(x), list(y)
            x.clear()
            y.clear()
        except StopIteration:
            break


def extract_with_overlap_of_n_minus_1_words_in_same_and_between_yields(file_obj, seq_len):
    """
    Given a file with words w1, w2, w3, ... wn and seq_len = k
    this func generates a tuple of two where the first element is a list of words
    and the second element is a shift of 1 of those words, the two elements have
    seq_len - 1 shared words.

    E.g given the raw text data:
    "WONDER HOW MUCH OF THE MEETINGS IS TALKING ABOUT THE STUFF AT THE MEETINGS </s>
    YEAH </s> NOT"
    and seq_len=4
    the function will yield (ignoring remainder):

    Yield #1:
      x = ['WONDER', 'HOW', 'MUCH', 'OF'], y = ['HOW', 'MUCH', 'OF', 'THE']
    Yield #2:
      x = ['MUCH', 'OF', 'THE', 'MEETINGS'], y = ['OF', 'THE', 'MEETINGS', 'IS']
    Yield #3:
      x = ['THE', 'MEETINGS', 'IS', 'TALKING'], y = ['MEETINGS', 'IS', 'TALKING', 'ABOUT']
    Yield #4:
      x = ['IS', 'TALKING', 'ABOUT', 'THE'], y = ['TALKING', 'ABOUT', 'THE', 'STUFF']
    Yield #5:
      x = ['ABOUT', 'THE', 'STUFF', 'AT'], y = ['THE', 'STUFF', 'AT', 'THE']
    Yield #6:
      x = ['STUFF', 'AT', 'THE', 'MEETINGS'], y = ['AT', 'THE', 'MEETINGS', '</s>']
    Yield #7:
      x = ['THE', 'MEETINGS', '</s>', 'YEAH'], y = ['MEETINGS', '</s>', 'YEAH', '</s>']

    Args:
        file_obj (file): opened file
        seq_len (int):

    Yields:
        pair of x and y that contains words shifted by 1

    """
    gen_words = _gen_n_words(file_obj=file_obj,
                             n=seq_len,
                             shift=True,
                             num_shifts=1)
    x = next(gen_words)
    y = next(gen_words)
    while True:
        try:
            yield list(x), list(y)
            # x = y since the words are shifted by 1 in time
            x = y
            y = next(gen_words)

        except StopIteration:
            break


def extract_x_without_overlap_y_shifted_by_1(file_obj, seq_len):
    """
    The classic approach is representing data to a language model.
    simply x contains words and y contains those words shifted by 1.
    No overlap between x from different yields.

    E.g given the raw text data:
    "WONDER HOW MUCH OF THE MEETINGS IS TALKING ABOUT THE STUFF AT THE MEETINGS </s>
    YEAH </s> NOT"
    and seq_len=4
    the function will yield (ignoring remainder):

    Yield #1:
      x=['WONDER', 'HOW', 'MUCH', 'OF'], y=['HOW', 'MUCH', 'OF', 'THE']
    Yield #2:
      x=['THE', 'MEETINGS', 'IS', 'TALKING'], y=['MEETINGS', 'IS', 'TALKING', 'ABOUT']
    Yield #3:
      x=['ABOUT', 'THE', 'STUFF', 'AT'], y=['THE', 'STUFF', 'AT', 'THE']

    Args:
        file_obj (file): opened file
        seq_len (int):

    Yields:
        pair of lists where the first is a list of words and the other
        a list of the next words

    """
    x = list()
    y = list()
    gen_w = _gen_next_word(file_obj=file_obj)

    while True:
        try:
            if len(x) != seq_len:
                x.append(next(gen_w))
            if len(x) == seq_len:
                y = x[1:]
                shared_w = next(gen_w)
                y.append(shared_w)
                yield list(x), list(y)
                x.clear()
                y.clear()
                x.append(shared_w)

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

    Yields:
        pair of x, y where x is list of words with a list size of seq_len
        and y is a list of part-of-speech tags of the words in x accordingly.
    """

    gen_words = _gen_n_words(file_obj=file_obj, n=seq_len)
    words = list()
    tags = list()
    for word, tag in gen_pos_dataset(gen_words):

        words.append(word)
        tags.append(tag)
        if len(words) == seq_len:
            yield words, tags
            words.clear()
            tags.clear()
