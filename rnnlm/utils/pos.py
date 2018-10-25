from nltk import pos_tag, sent_tokenize, word_tokenize
from collections import defaultdict


def create_pos_dataset(output_file, gen_words):
    """
    creates POS dataset in CSV format
    Args:
        output_file:
        gen_words:

    Returns:
        None - file is writen to disk
    """
    num_of_iterations = 0
    count_diff_tags = defaultdict(int)
    csv_file = open(output_file, 'w')
    csv_file.write("x,y\n")

    for words_list, _ in gen_words:
        words_without_tags = list()

        # deal with special tags such as <oos>, <unk> and </s>
        # chars like < and > messes with the tagging therefore they are removed
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

        words_str = " ".join(words_without_tags)
        sentences = sent_tokenize(words_str)
        tagged_sentences = list()
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tagged_sentences.append(pos_tag(tokens))

            for tagged_sentence in tagged_sentences:
                for word, tag in tagged_sentence:
                    csv_file.write("{},{}\n".format(word, tag))
                    count_diff_tags[tag] += 1

        num_of_iterations += 1

    print("num of diff tags = {}".format(len(count_diff_tags)))
    print("tags = {}".format(count_diff_tags))
    print("num of iterations = {}".format(num_of_iterations))
