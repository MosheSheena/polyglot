from rnnlm.models.lstm_fast.model import create_model
from rnnlm.models.lstm_fast.loss import create_loss
from rnnlm.models.lstm_fast.optimizer import create_optimizer
from rnnlm.utils.estimator.estimator import train_and_evaluate_model
from rnnlm.utils.estimator.estimator_hook.learning_rate_decay import LearningRateDecayHook
from rnnlm.utils.estimator.estimator_hook.init_legacy_model import InitLegacyModelHook

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

    # TODO support easy multitask and transfer learning training
    print("Start training")
    # print("training pos classifier")
    # train_and_evaluate_model(create_model=create_model,
    #                          create_loss=create_loss,
    #                          create_optimizer=create_optimizer,
    #                          train_tf_record_path=pos_train_tf_record_path,
    #                          valid_tf_record_path=pos_valid_tf_record_path,
    #                          test_tf_record_path=pos_test_tf_record_path,
    #                          num_epochs=hyperparams.train.num_epochs,
    #                          epoch_size_train=hyperparams.train.epoch_size_train_pos,
    #                          epoch_size_valid=hyperparams.train.epoch_size_valid_pos,
    #                          epoch_size_test=hyperparams.train.epoch_size_test_pos,
    #                          hyperparams=hyperparams,
    #                          checkpoint_path=abs_save_path,
    #                          training_hooks=[LearningRateDecayHook])
    LearningRateDecayHook.epoch_counter = 0

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
                             checkpoint_path=abs_save_path,
                             training_hooks=[LearningRateDecayHook, InitLegacyModelHook],
                             evaluation_hooks=[InitLegacyModelHook])

    # Reset epoch counter for other train sessions
    # for supporting Transfer Learning or MultiTask Learning
    # TODO manage this counter somewhere else to support multitask
    LearningRateDecayHook.epoch_counter = 0

    print("End training")
