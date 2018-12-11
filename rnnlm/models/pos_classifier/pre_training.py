from rnnlm.utils.tf_io import io_service, extractor
from rnnlm.utils.preprocessor import preprocess
import os
import tensorflow as tf


def preprocess_elements_with_vocab(gen_fn,
                                   seq_len,
                                   abs_vocab_path_features,
                                   abs_vocab_path_labels,
                                   abs_raw_data_train,
                                   abs_raw_data_valid,
                                   abs_raw_data_test,
                                   abs_train_tf_record_path,
                                   abs_valid_tf_record_path,
                                   abs_test_tf_record_path):
    """
    Enumerates every single element in the data by its appearance in the vocab
    Args:
        gen_fn (func): function that yields the features and labels from raw data
        seq_len (int):
        abs_vocab_path_features (str): absolute path of the vocab for features
        abs_vocab_path_labels (str): absolute path of the vocab for labels
        abs_raw_data_train (str): absolute path of the original training data
        abs_raw_data_valid (str): absolute path of the original validation data
        abs_raw_data_test (str): absolute path of the original test data
        abs_train_tf_record_path (str): absolute path of output file for train
        abs_valid_tf_record_path (str): absolute path of output file for validation
        abs_test_tf_record_path (str): absolute path of output file for test

    Returns:
        None
    """
    vocab_features = preprocess.build_vocab(abs_vocab_path_features)
    vocab_labels = preprocess.build_vocab(abs_vocab_path_labels)
    with tf.gfile.GFile(abs_raw_data_train, 'r') as f:
        io_service.raw_to_tf_records(gen_raw_data=gen_fn(f, seq_len),
                                     abs_tf_record_path=abs_train_tf_record_path,
                                     preprocessor_feature_fn=preprocess.map_elements_to_ids,
                                     preprocessor_feature_params=vocab_features,
                                     preprocessor_label_fn=preprocess.map_elements_to_ids,
                                     preprocessor_label_params=vocab_labels)
    with tf.gfile.GFile(abs_raw_data_valid, 'r') as f:
        io_service.raw_to_tf_records(gen_raw_data=gen_fn(f, seq_len),
                                     abs_tf_record_path=abs_valid_tf_record_path,
                                     preprocessor_feature_fn=preprocess.map_elements_to_ids,
                                     preprocessor_feature_params=vocab_features,
                                     preprocessor_label_fn=preprocess.map_elements_to_ids,
                                     preprocessor_label_params=vocab_labels)
    with tf.gfile.GFile(abs_raw_data_test, 'r') as f:
        io_service.raw_to_tf_records(gen_raw_data=gen_fn(f, seq_len),
                                     abs_tf_record_path=abs_test_tf_record_path,
                                     preprocessor_feature_fn=preprocess.map_elements_to_ids,
                                     preprocessor_feature_params=vocab_features,
                                     preprocessor_label_fn=preprocess.map_elements_to_ids,
                                     preprocessor_label_params=vocab_labels)


def main(hyperparams):
    abs_data_path = os.path.join(os.getcwd(), hyperparams.problem.data_path)
    abs_vocab_path = os.path.join(os.getcwd(), hyperparams.problem.vocab_path)
    abs_pos_vocab_path = os.path.join(os.getcwd(), hyperparams.problem.pos_vocab_path)
    abs_tf_record_path = os.path.join(os.getcwd(), hyperparams.problem.tf_records_path)

    train_tf_record_path = os.path.join(abs_tf_record_path, "train.tfrecord")
    valid_tf_record_path = os.path.join(abs_tf_record_path, "valid.tfrecord")
    test_tf_record_path = os.path.join(abs_tf_record_path, "test.tfrecord")

    pos_train_tf_record_path = os.path.join(abs_tf_record_path, "train_pos.tfrecord")
    pos_valid_tf_record_path = os.path.join(abs_tf_record_path, "valid_pos.tfrecord")
    pos_test_tf_record_path = os.path.join(abs_tf_record_path, "test_pos.tfrecord")

    if not os.path.exists(abs_tf_record_path):
        os.makedirs(abs_tf_record_path)

    print("Converting raw data to tfrecord format")

    # preprocess for classic training
    print("converting original data to tf record")
    preprocess_elements_with_vocab(gen_fn=extractor.gen_no_overlap_words,
                                   seq_len=hyperparams.arch.sequence_length,
                                   abs_vocab_path_features=abs_vocab_path,
                                   abs_vocab_path_labels=abs_vocab_path,
                                   abs_raw_data_train=os.path.join(abs_data_path, "train"),
                                   abs_raw_data_valid=os.path.join(abs_data_path, "valid"),
                                   abs_raw_data_test=os.path.join(abs_data_path, "test"),
                                   abs_train_tf_record_path=train_tf_record_path,
                                   abs_valid_tf_record_path=valid_tf_record_path,
                                   abs_test_tf_record_path=test_tf_record_path)

    # preprocess for pos training
    # print("converting pos tf records")
    # preprocess_elements_with_vocab(gen_fn=generator.gen_pos_tagger,
    #                                seq_len=hyperparams.arch.sequence_length,
    #                                abs_vocab_path_features=abs_vocab_path,
    #                                abs_vocab_path_labels=abs_pos_vocab_path,
    #                                abs_raw_data_train=os.path.join(abs_data_path, "train"),
    #                                abs_raw_data_valid=os.path.join(abs_data_path, "valid"),
    #                                abs_raw_data_test=os.path.join(abs_data_path, "test"),
    #                                abs_train_tf_record_path=pos_train_tf_record_path,
    #                                abs_valid_tf_record_path=pos_valid_tf_record_path,
    #                                abs_test_tf_record_path=pos_test_tf_record_path)
    print("Conversion done.")
