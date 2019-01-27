from rnnlm.utils.tf_io import extractor
from rnnlm.utils.tf_io.preprocessor.preprocess import preprocess_elements_with_vocab

import os


def main(shared_hyperparams, hyperparams):

    abs_data_path = os.path.join(os.getcwd(), hyperparams.problem.data_path)
    abs_vocab_path = os.path.join(os.getcwd(), hyperparams.problem.vocab_path)
    abs_gen_vocab_path = os.path.join(os.getcwd(), hyperparams.problem.gen_vocab)
    abs_tf_record_path = os.path.join(os.getcwd(), shared_hyperparams.problem.tf_records_path)

    train_raw_data_path = os.path.join(abs_data_path, hyperparams.problem.train_raw_data_file)
    valid_raw_data_path = os.path.join(abs_data_path, hyperparams.problem.valid_raw_data_file)
    test_raw_data_path = os.path.join(abs_data_path, hyperparams.problem.test_raw_data_file)

    train_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.problem.tf_record_train_file)
    valid_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.problem.tf_record_valid_file)
    test_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.problem.tf_record_test_file)

    raw_files = [train_raw_data_path, valid_raw_data_path, test_raw_data_path]
    tf_record_outputs = [train_tf_record_path, valid_tf_record_path, test_tf_record_path]

    if not os.path.exists(abs_tf_record_path):
        os.makedirs(abs_tf_record_path)

    # preprocess for classic training
    print("converting original data to tf record")
    seq_len = shared_hyperparams.arch.sequence_length
    x_shifts = hyperparams.problem.get_or_default(key="num_shifts_x", default=seq_len)
    gen_fn = extractor.extract_x_y_words_with_x_shifting_by_n_each_yield

    for raw_path, tf_record_path in zip(raw_files, tf_record_outputs):
        with open(raw_path, 'r') as f:

            # TODO - change below
            preprocess_elements_with_vocab(gen_fn=gen_fn(file_obj=f, seq_len=seq_len, n=x_shifts),
                                           abs_vocab_path_features=abs_vocab_path,
                                           abs_vocab_path_labels=abs_gen_vocab_path,
                                           abs_output_tf_record_path=tf_record_path)

