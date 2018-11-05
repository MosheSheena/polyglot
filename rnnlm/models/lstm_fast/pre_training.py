from rnnlm.models.lstm_fast import io_service
from rnnlm.utils.preprocessor import preprocess
import os


def main(hyperparams):
    abs_data_path = os.path.join(os.getcwd(), hyperparams.problem.data_path)
    abs_vocab_path = os.path.join(os.getcwd(), hyperparams.problem.vocab_path)
    abs_save_path = os.path.join(os.getcwd(), hyperparams.train.save_path)
    abs_tf_record_path = os.path.join(os.getcwd(), hyperparams.problem.tf_records_path)

    train_tf_record_path = os.path.join(abs_tf_record_path, "train.tfrecord")
    valid_tf_record_path = os.path.join(abs_tf_record_path, "valid.tfrecord")
    test_tf_record_path = os.path.join(abs_tf_record_path, "test.tfrecord")

    if not os.path.exists(abs_tf_record_path):
        os.makedirs(abs_tf_record_path)

    print("Converting raw data to tfrecord format")
    vocab = preprocess.build_vocab(abs_vocab_path)
    io_service.raw_to_tf_records(raw_path=os.path.join(abs_data_path, "train"),
                                 tf_record_path=train_tf_record_path,
                                 seq_len=hyperparams.arch.hidden_layer_depth,
                                 preprocessor_feature_fn=preprocess.map_elements_to_ids,
                                 preprocessor_feature_params=vocab,
                                 preprocessor_label_fn=preprocess.map_elements_to_ids,
                                 preprocessor_label_params=vocab)
    io_service.raw_to_tf_records(raw_path=os.path.join(abs_data_path, "valid"),
                                 tf_record_path=valid_tf_record_path,
                                 seq_len=hyperparams.arch.hidden_layer_depth,
                                 preprocessor_feature_fn=preprocess.map_elements_to_ids,
                                 preprocessor_feature_params=vocab,
                                 preprocessor_label_fn=preprocess.map_elements_to_ids,
                                 preprocessor_label_params=vocab)
    io_service.raw_to_tf_records(raw_path=os.path.join(abs_data_path, "test"),
                                 tf_record_path=test_tf_record_path,
                                 seq_len=hyperparams.arch.hidden_layer_depth,
                                 preprocessor_feature_fn=preprocess.map_elements_to_ids,
                                 preprocessor_feature_params=vocab,
                                 preprocessor_label_fn=preprocess.map_elements_to_ids,
                                 preprocessor_label_params=vocab)
    print("Conversion done.")
