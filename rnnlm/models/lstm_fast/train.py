from rnnlm.models.lstm_fast.model import create_model
from rnnlm.models.lstm_fast.loss import create_loss
from rnnlm.models.lstm_fast.optimizer import create_optimizer
from rnnlm.utils.estimator import train_and_evaluate_model
import os


def main(hyperparams):

    abs_save_path = os.path.join(os.getcwd(), hyperparams.train.save_path)
    abs_tf_record_path = os.path.join(os.getcwd(), hyperparams.problem.tf_records_path)

    train_tf_record_path = os.path.join(abs_tf_record_path, "train.tfrecord")
    valid_tf_record_path = os.path.join(abs_tf_record_path, "valid.tfrecord")
    test_tf_record_path = os.path.join(abs_tf_record_path, "test.tfrecord")

    print("Start training")

    train_and_evaluate_model(create_model=create_model,
                             create_loss=create_loss,
                             create_optimizer=create_optimizer,
                             hyperparams=hyperparams,
                             train_tf_record_path=train_tf_record_path,
                             valid_tf_record_path=valid_tf_record_path,
                             test_tf_record_path=test_tf_record_path)

    print("End training")
