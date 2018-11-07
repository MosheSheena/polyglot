from nltk import pos_tag, sent_tokenize, word_tokenize
from collections import defaultdict


def get_words_without_tags(words_list):
    """
    deal with special tags such as <oos>, <unk> and </s>
    chars like < and > messes with the tagging therefore they are removed

    Returns:
        list of words without special tags
    """
    words_without_tags = list()

    previous_word = ""
    for w in words_list:
        if w.startswith('<') and w.endswith('>'):
            if w == '</s>':
                if previous_word != "":
                    previous_word += '.'
                    words_without_tags.pop()
                    words_without_tags.append(previous_word)
                else:
                    previous_word = '.'
                continue
            else:
                w = w[1:-1]
        words_without_tags.append(w)
        previous_word = w

    return words_without_tags


def gen_pos_dataset(gen_words):
    """
    generator function that yields a tuple of a list of words from
     gen_words, which is also a generator, and the POS tags of those words.
    Args:
        gen_words (generator func):

    Yields:
        tuple (word, pos_tag)
    """
    num_of_iterations = 0
    count_diff_tags = defaultdict(int)

    for words_list, _ in gen_words:

        words_without_tags = get_words_without_tags(words_list)
        words_str = " ".join(words_without_tags)
        sentences = sent_tokenize(words_str)
        tagged_sentences = list()
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tagged_sentences.append(pos_tag(tokens))

            for tagged_sentence in tagged_sentences:
                for word, tag in tagged_sentence:
                    count_diff_tags[tag] += 1
                    yield word, tag

        num_of_iterations += 1

    print("num of diff tags = {}".format(len(count_diff_tags)))
    print("tags = {}".format(count_diff_tags))
    print("num of iterations = {}".format(num_of_iterations))
