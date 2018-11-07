from rnnlm.models.lstm_fast.model import create_model
from rnnlm.models.lstm_fast.loss import create_loss
from rnnlm.models.lstm_fast.optimizer import create_optimizer
from rnnlm.utils.estimator.estimator import train_and_evaluate_model
import os


def main(hyperparams):

    abs_save_path = os.path.join(os.getcwd(), hyperparams.problem.save_path)
    abs_tf_record_path = os.path.join(os.getcwd(), hyperparams.problem.tf_records_path)

    train_tf_record_path = os.path.join(abs_tf_record_path, "train.tfrecord")
    valid_tf_record_path = os.path.join(abs_tf_record_path, "valid.tfrecord")
    test_tf_record_path = os.path.join(abs_tf_record_path, "test.tfrecord")

    pos_train_tf_record_path = os.path.join(abs_tf_record_path, "train_pos.tfrecord")
    pos_valid_tf_record_path = os.path.join(abs_tf_record_path, "valid_pos.tfrecord")
    pos_test_tf_record_path = os.path.join(abs_tf_record_path, "test_pos.tfrecord")

    print("Start training")
    print("training pos classifier")
    train_and_evaluate_model(create_model=create_model,
                             create_loss=create_loss,
                             create_optimizer=create_optimizer,
                             train_tf_record_path=pos_train_tf_record_path,
                             valid_tf_record_path=pos_valid_tf_record_path,
                             test_tf_record_path=pos_test_tf_record_path,
                             num_epochs=hyperparams.train.num_epochs,
                             epoch_size_train=1406,
                             epoch_size_valid=144,
                             epoch_size_test=178,
                             hyperparams=hyperparams,
                             checkpoint_path=abs_save_path)

    print("training language model")
    train_and_evaluate_model(create_model=create_model,
                             create_loss=create_loss,
                             create_optimizer=create_optimizer,
                             train_tf_record_path=train_tf_record_path,
                             valid_tf_record_path=valid_tf_record_path,
                             test_tf_record_path=test_tf_record_path,
                             num_epochs=hyperparams.train.num_epochs,
                             epoch_size_train=hyperparams.train.epoch_size_train,
                             epoch_size_valid=hyperparams.train.epoch_size_valid,
                             epoch_size_test=hyperparams.train.epoch_size_test,
                             hyperparams=hyperparams,
                             checkpoint_path=abs_save_path)

    print("End training")
