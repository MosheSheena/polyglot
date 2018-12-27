from rnnlm.utils.tf_io import extractor
from rnnlm.utils.tf_io.preprocessor.preprocess import preprocess_elements_with_vocab

import os


def main(shared_hyperparams, hyperparams):

    abs_data_path = os.path.join(os.getcwd(), hyperparams.problem.data_path)
    abs_vocab_path = os.path.join(os.getcwd(), hyperparams.problem.vocab_path)
    abs_tf_record_path = os.path.join(os.getcwd(), shared_hyperparams.problem.tf_records_path)

    train_raw_data_path = os.path.join(abs_data_path, hyperparams.problem.train_raw_data_file)
    valid_raw_data_path = os.path.join(abs_data_path, hyperparams.problem.valid_raw_data_file)
    test_raw_data_path = os.path.join(abs_data_path, hyperparams.problem.test_raw_data_file)

    train_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.problem.tf_record_train_file)
    valid_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.problem.tf_record_valid_file)
    test_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.problem.tf_record_test_file)

    if not os.path.exists(abs_tf_record_path):
        os.makedirs(abs_tf_record_path)

    # preprocess for classic training
    print("converting original data to tf record")

    preprocess_elements_with_vocab(gen_fn=extractor.extract_x_without_overlap_y_shifted_by_1,
                                   seq_len=shared_hyperparams.arch.sequence_length,
                                   abs_vocab_path_features=abs_vocab_path,
                                   abs_vocab_path_labels=abs_vocab_path,
                                   abs_raw_data_train=train_raw_data_path,
                                   abs_raw_data_valid=valid_raw_data_path,
                                   abs_raw_data_test=test_raw_data_path,
                                   abs_train_tf_record_path=train_tf_record_path,
                                   abs_valid_tf_record_path=valid_tf_record_path,
                                   abs_test_tf_record_path=test_tf_record_path)

