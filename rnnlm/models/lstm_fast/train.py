import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.INFO)

from time import gmtime, strftime
import os

from rnnlm.utils.hyperparams import load_params
from rnnlm.models.lstm_fast.model import create_model
from rnnlm.models.lstm_fast.loss import create_loss
from rnnlm.models.lstm_fast.optimizer import create_optimizer
from rnnlm.models.lstm_fast import io_service
from rnnlm.models.lstm_fast.estimator import train_and_evaluate_model


def main():
    hyperparams = load_params(os.path.join(os.getcwd(), "rnnlm/models/lstm_fast/hyperparameters.json"))
    print(strftime("start time: %Y-%m-%d %H:%M:%S", gmtime()))

    if not hyperparams.problem.data_path:
        raise ValueError("Must set data_path hyperparameters.json")

    abs_data_path = os.path.join(os.getcwd(), hyperparams.problem.data_path)
    abs_vocab_path = os.path.join(os.getcwd(), hyperparams.problem.vocab_path)
    abs_save_path = os.path.join(os.getcwd(), hyperparams.train.save_path)
    abs_tf_record_path = os.path.join(os.getcwd(), hyperparams.problem.tf_records_path)

    train_tf_record_path = os.path.join(abs_tf_record_path, "train.tfrecord")
    valid_tf_record_path = os.path.join(abs_tf_record_path, "valid.tfrecord")
    test_tf_record_path = os.path.join(abs_tf_record_path, "test.tfrecord")

    if not os.path.exists(abs_tf_record_path):
        os.makedirs(abs_tf_record_path)

    if hyperparams.problem.convert_raw_to_tf_records:
        print("Converting raw data to tfrecord format")
        io_service.raw_to_tf_records(raw_path=os.path.join(abs_data_path, "train"),
                                     tf_record_path=train_tf_record_path,
                                     vocab_path=abs_vocab_path,
                                     seq_len=hyperparams.arch.hidden_layer_depth)
        io_service.raw_to_tf_records(raw_path=os.path.join(abs_data_path, "valid"),
                                     tf_record_path=valid_tf_record_path,
                                     vocab_path=abs_vocab_path,
                                     seq_len=hyperparams.arch.hidden_layer_depth)
        io_service.raw_to_tf_records(raw_path=os.path.join(abs_data_path, "test"),
                                     tf_record_path=test_tf_record_path,
                                     vocab_path=abs_vocab_path,
                                     seq_len=hyperparams.arch.hidden_layer_depth)
        print("Conversion done.")

    print("Start training")

    train_and_evaluate_model(create_model=create_model,
                             create_loss=create_loss,
                             create_optimizer=create_optimizer,
                             hyperparams=hyperparams)

    print("End training")

    print(strftime("end time: %Y-%m-%d %H:%M:%S", gmtime()))


if __name__ == "__main__":

    main()
