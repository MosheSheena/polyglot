import tensorflow as tf

from rnnlm.models.lstm_fast import reader
from rnnlm.models.lstm_fast import writer


def _words_to_ids(words, vocab):

    return [vocab[word] if word in vocab else vocab["<oos>"] for word in words]


def raw_to_tf_records(raw_path, tf_record_path, vocab_path, seq_len, overlap=False):
    """
    convert raw data (sentences) into tf records format
    Args:
        raw_path: (str)
        tf_record_path: (str)
        vocab_path: (str) path of raw format vocabulary
        seq_len: (int)
        overlap: (bool) whether the data should written and then read with overlaps.
            e.g. suppose the sentence "The quick brown fox jumped over the lazy dog"
            given seq_len=2 and say batch_size=2 we will get the following:
            x1=[[The, quick], [quick, brown]], x2=[[brown, fox], [fox, jumped]]
            y1=[[quick, brown], [brown, fox]], y2=[[fox, jumped], [jumped, over]]
            notice the overlap for example the word quick that appears in x11 and x12
            and the overlap between batches (the word brown appears in x1 and x2)
            y is represented accordingly only shifted by 1
    Returns:
        None
    """
    raw_file = tf.gfile.GFile(raw_path, "r")
    vocab_file = tf.gfile.GFile(vocab_path, "r")
    vocab = reader.build_vocab(vocab_file)
    packed_vocab = [vocab]

    if overlap:
        gen_words = reader.gen_shifted_words_with_overlap(file_obj=raw_file, seq_len=seq_len)
    else:
        gen_words = reader.gen_no_overlap_words(file_obj=raw_file, seq_len=seq_len)

    writer.write_tf_records(gen_words=gen_words,
                            destination_path=tf_record_path,
                            preprocessor_feature_fn=_words_to_ids,
                            preprocessor_feature_params=packed_vocab,
                            preprocessor_label_fn=_words_to_ids,
                            preprocessor_label_params=packed_vocab)


def load_tf_records(tf_record_path, batch_size, seq_len):
    return reader.read_tf_records(tf_record_path=tf_record_path, batch_size=batch_size, seq_len=seq_len)
