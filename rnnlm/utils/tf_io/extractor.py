import random

from rnnlm.utils.tf_io.pos import gen_pos_dataset


def _gen_next_word(opened_file):
    """
    Yields all the words from an opened file, one by one, without loading line by line
    Args:
        opened_file: an opened file

    Yields:
        the next word from the file
    """
    last = ""
    while True:
        buf = opened_file.read(3000)
        if not buf:
            break
        words = (last + buf).split()
        last = words.pop()
        for word in words:
            yield word
    if last:
        yield last


def _gen_n_words(opened_file,
                 n,
                 shift=False,
                 num_shifts=0):
    """
    Generates n words from file. if shift flag is true than num_shifts arg is
    expected, the function will yield words shifted by num_shifts.
    Examples: given the raw text data:
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
        opened_file (file): opened file
        n (int): num of words to read from file
        shift (bool): whether to yield shifted words from file instead on yielding
         n different each time
        num_shifts (int): how many elements to shift between each yield

    Yields:
        list of n words from file
    """

    n_words = list()

    gen_read_words = _gen_next_word(opened_file)
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


def extract_x_y_words_with_x_shifting_by_n_each_yield(opened_file, seq_len, n):
    """
    extracts x with a shift of n each yield, y is always extracted with a shift of
    1 relative to x
    Examples:
    given the raw text data:
    "WONDER HOW MUCH OF THE MEETINGS IS TALKING ABOUT THE STUFF AT THE MEETINGS </s>
    YEAH </s>"
    and seq_len=4, n=2
    the function will yield (ignoring remainder):

    Yield #1:
      x = ['WONDER', 'HOW', 'MUCH', 'OF'], y = ['HOW', 'MUCH', 'OF', 'THE']
    Yield #2:
      x = ['MUCH', 'OF', 'THE', 'MEETINGS'], y = ['OF', 'THE', 'MEETINGS', 'IS']
    Yield #3:
      x = ['THE', 'MEETINGS', 'IS', 'TALKING'], y = ['MEETINGS', 'IS', 'TALKING', 'ABOUT']
    Yield #4:
      x = ['IS', 'TALKING', 'ABOUT', 'THE'], y = ['TALKING', 'ABOUT', 'THE', 'AT']
    Yield #5:
      x = ['ABOUT', 'THE', 'AT', 'THE'], y = ['THE', 'AT', 'THE', 'MEETINGS']
    Yield #6:
      x = ['AT', 'THE', 'MEETINGS', '</s>], y = ['THE', 'MEETINGS', '</s>', 'YEAH']


    Args:
        opened_file : an opened file that has the raw data
        seq_len (int): the sequence length of the yield
        n (int): num of shifts to perform on x

    Yields:
        tuple of 2 containing x and y.
    """
    assert n <= seq_len, "cannot overlap more than sequence length {}".format(seq_len)
    shifter = _gen_n_words(opened_file=opened_file, n=1)

    accumulator = list()
    while True:
        try:
            while len(accumulator) < seq_len:
                accumulator += next(shifter)
            x = list(accumulator)
            accumulator.pop(0)
            accumulator += next(shifter)
            y = list(accumulator)
            yield x, y
            for i in range(n - 1):
                accumulator.pop(0)
        except StopIteration:
            break


def extract_without_overlap_in_same_and_between_yields(opened_file, seq_len):
    """
    Generates x, y where each element is shifted by seq_len
    Examples:
    given the raw text data:
    "WONDER HOW MUCH OF THE MEETINGS IS TALKING ABOUT THE STUFF AT THE MEETINGS </s>
    YEAH </s>"
    and seq_len=4
    the function will yield (ignoring remainder):

    Yield #1:
      x = ['WONDER', 'HOW', 'MUCH', 'OF'], y = ['THE', 'MEETINGS', 'IS', 'TALKING']
    Yield #2:
      x = ['ABOUT', 'THE', 'STUFF', 'AT'], y = ['THE', 'MEETINGS' '</s>', 'YEAH']

    Args:
        opened_file (file): opened file
        seq_len (int):

    Yields:
    tuple of x and y.
    """
    gen_w = _gen_n_words(opened_file=opened_file, n=seq_len)
    while True:
        try:
            x = next(gen_w)
            y = next(gen_w)
            yield list(x), list(y)
            x.clear()
            y.clear()
        except StopIteration:
            break


def extract_with_overlap_of_n_minus_1_words_in_same_and_between_yields(opened_file, seq_len):
    """
    Given a file with words w1, w2, w3, ... wn and seq_len = k
    this func generates a tuple of two where the first element is a list of words
    and the second element is a shift of 1 of those words, the two elements have
    seq_len - 1 shared words.

    Examples:
    given the raw text data:
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
        opened_file (file): opened file
        seq_len (int):

    Yields:
        pair of x and y that contains words shifted by 1

    """
    gen_words = _gen_n_words(opened_file=opened_file,
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


def extract_x_without_overlap_y_shifted_by_1(opened_file, seq_len):
    """
    The classic approach is representing data to a language model.
    simply x contains words and y contains those words shifted by 1.
    No overlap between x from different yields.

    Examples:
    given the raw text data:
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
        opened_file (file): opened file
        seq_len (int):

    Yields:
        pair of lists where the first is a list of words and the other
        a list of the next words

    """
    x = list()
    y = list()
    gen_w = _gen_next_word(opened_file=opened_file)

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


def extract_words_and_their_pos_tags(opened_file, seq_len):
    """
    Each call to this generator generates a tuple of x, y
    where x is list of words with a list size of seq_len
    and y is a list of part-of-speech tags of the words
    in x accordingly.
    Args:
        opened_file: opened file of raw data, we  a
        seq_len: the length of how much we will read from the
            file in each generation

    Yields:
        pair of x, y where x is list of words with a list size of seq_len
        and y is a list of part-of-speech tags of the words in x accordingly.
    """

    gen_words = _gen_n_words(opened_file=opened_file, n=seq_len)
    words = list()
    tags = list()
    for word, tag in gen_pos_dataset(gen_words):

        words.append(word)
        tags.append(tag)
        if len(words) == seq_len:
            yield words, tags
            words.clear()
            tags.clear()


def extract_real_with_generated_dataset(seq_len,
                                        real_dataset,
                                        num_shifts_real,
                                        generated_dataset,
                                        num_shift_generated):
    real_tag = 'REAL_SENTENCE'
    generated_tag = 'GENERATED_SENTENCE'
    gen_words_1 = _gen_n_words(real_dataset, n=seq_len, num_shifts=num_shifts_real)
    gen_words_2 = _gen_n_words(generated_dataset, n=seq_len, num_shifts=num_shift_generated)

    has_real = True
    has_gen = True

    while has_real or has_gen:
        choice = random.randrange(0, 2)
        if choice == 0 and has_real:
            try:
                x = next(gen_words_1)
                yield list(x), list([real_tag])
            except StopIteration:
                has_real = False
        else:
            try:
                x = next(gen_words_2)
                yield list(x), list([generated_tag])
            except StopIteration:
                has_gen = False

