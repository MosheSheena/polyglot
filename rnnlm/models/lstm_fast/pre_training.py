from rnnlm.utils.tf_io import io_service
from rnnlm.utils.preprocessor import preprocess
import os


def preprocess_elements_with_vocab(abs_vocab_path_features,
                                   abs_vocab_path_labels,
                                   abs_raw_data_train,
                                   abs_raw_data_valid,
                                   abs_raw_data_test,
                                   abs_train_tf_record_path,
                                   abs_valid_tf_record_path,
                                   abs_test_tf_record_path,
                                   seq_len):
    """
    Enumerates every single element in the data by its appearance in the vocab
    Args:
        abs_vocab_path_features (str): absolute path of the vocab for features
        abs_vocab_path_labels (str): absolute path of the vocab for labels
        abs_raw_data_train (str): absolute path of the original training data
        abs_raw_data_valid (str): absolute path of the original validation data
        abs_raw_data_test (str): absolute path of the original test data
        abs_train_tf_record_path (str): absolute path of output file for train
        abs_valid_tf_record_path (str): absolute path of output file for validation
        abs_test_tf_record_path (str): absolute path of output file for test
        seq_len (int):

    Returns:

    """
    vocab_features = preprocess.build_vocab(abs_vocab_path_features)
    vocab_labels = preprocess.build_vocab(abs_vocab_path_labels)
    io_service.raw_to_tf_records(raw_path=abs_raw_data_train,
                                 abs_tf_record_path=abs_train_tf_record_path,
                                 seq_len=seq_len,
                                 preprocessor_feature_fn=preprocess.map_elements_to_ids,
                                 preprocessor_feature_params=vocab_features,
                                 preprocessor_label_fn=preprocess.map_elements_to_ids,
                                 preprocessor_label_params=vocab_labels)
    io_service.raw_to_tf_records(raw_path=abs_raw_data_valid,
                                 abs_tf_record_path=abs_valid_tf_record_path,
                                 seq_len=seq_len,
                                 preprocessor_feature_fn=preprocess.map_elements_to_ids,
                                 preprocessor_feature_params=vocab_features,
                                 preprocessor_label_fn=preprocess.map_elements_to_ids,
                                 preprocessor_label_params=vocab_labels)
    io_service.raw_to_tf_records(raw_path=abs_raw_data_test,
                                 abs_tf_record_path=abs_test_tf_record_path,
                                 seq_len=seq_len,
                                 preprocessor_feature_fn=preprocess.map_elements_to_ids,
                                 preprocessor_feature_params=vocab_features,
                                 preprocessor_label_fn=preprocess.map_elements_to_ids,
                                 preprocessor_label_params=vocab_labels)


def main(hyperparams):
    abs_data_path = os.path.join(os.getcwd(), hyperparams.problem.data_path)
    abs_vocab_path = os.path.join(os.getcwd(), hyperparams.problem.vocab_path)
    # abs_pos_vocab_path = os.path.join(os.getcwd(), hyperparams.problem.pos_vocab_path)
    abs_save_path = os.path.join(os.getcwd(), hyperparams.train.save_path)
    abs_tf_record_path = os.path.join(os.getcwd(), hyperparams.problem.tf_records_path)

    train_tf_record_path = os.path.join(abs_tf_record_path, "train.tfrecord")
    valid_tf_record_path = os.path.join(abs_tf_record_path, "valid.tfrecord")
    test_tf_record_path = os.path.join(abs_tf_record_path, "test.tfrecord")

    if not os.path.exists(abs_tf_record_path):
        os.makedirs(abs_tf_record_path)

    print("Converting raw data to tfrecord format")

    # preprocess for classic training
    preprocess_elements_with_vocab(abs_vocab_path_features=abs_vocab_path,
                                   abs_vocab_path_labels=abs_vocab_path,
                                   abs_raw_data_train=os.path.join(abs_data_path, "train"),
                                   abs_raw_data_valid=os.path.join(abs_data_path, "valid"),
                                   abs_raw_data_test=os.path.join(abs_data_path, "test"),
                                   abs_train_tf_record_path=train_tf_record_path,
                                   abs_valid_tf_record_path=valid_tf_record_path,
                                   abs_test_tf_record_path=test_tf_record_path,
                                   seq_len=hyperparams.arch.hidden_layer_depth)

    # preprocess for pos training
    # preprocess_elements_with_vocab(abs_vocab_path_features=abs_vocab_path,
    #                                abs_vocab_path_labels=abs_pos_vocab_path,
    #                                abs_raw_data_train=os.path.join(abs_data_path, "pos_train"),
    #                                abs_raw_data_valid=os.path.join(abs_data_path, "pos_valid"),
    #                                abs_raw_data_test=os.path.join(abs_data_path, "pos_test"),
    #                                abs_train_tf_record_path=train_tf_record_path,
    #                                abs_valid_tf_record_path=valid_tf_record_path,
    #                                abs_test_tf_record_path=test_tf_record_path,
    #                                seq_len=hyperparams.arch.hidden_layer_depth)
    print("Conversion done.")
