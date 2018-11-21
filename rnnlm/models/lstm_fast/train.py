from rnnlm.models.lstm_fast.model import create_model
from rnnlm.models.lstm_fast.loss import create_loss
from rnnlm.models.lstm_fast.optimizer import create_optimizer
from rnnlm.utils.estimator.estimator_hook.learning_rate_decay import LearningRateDecayHook
from rnnlm.utils.estimator.estimator_hook.init_legacy_model import InitLegacyModelHook
from rnnlm.utils.trainer import trainer
from rnnlm.classes.task import Task
import os


def main(hyperparams):
    abs_save_path = os.path.join(os.getcwd(), hyperparams.problem.save_path)
    abs_tf_record_path = os.path.join(os.getcwd(), hyperparams.problem.tf_records_path)

    train_tf_record_path = os.path.join(abs_tf_record_path, "train.tfrecord")
    valid_tf_record_path = os.path.join(abs_tf_record_path, "valid.tfrecord")
    test_tf_record_path = os.path.join(abs_tf_record_path, "test.tfrecord")

    # train_tf_record_path = os.path.join(abs_tf_record_path, "train_legacy_reader.tfrecord")
    # valid_tf_record_path = os.path.join(abs_tf_record_path, "valid_legacy_reader.tfrecord")
    # test_tf_record_path = os.path.join(abs_tf_record_path, "test_legacy_reader.tfrecord")

    pos_train_tf_record_path = os.path.join(abs_tf_record_path, "train_pos.tfrecord")
    pos_valid_tf_record_path = os.path.join(abs_tf_record_path, "valid_pos.tfrecord")
    pos_test_tf_record_path = os.path.join(abs_tf_record_path, "test_pos.tfrecord")

    print("Start training")
    pos_classifier = Task(name="pos_classifier",
                          create_model=create_model,
                          create_loss=create_loss,
                          create_optimizer=create_optimizer,
                          train_tf_record_path=pos_train_tf_record_path,
                          valid_tf_record_path=pos_valid_tf_record_path,
                          test_tf_record_path=pos_test_tf_record_path,
                          hyperparams=hyperparams,
                          checkpoint_path=abs_save_path)

    lstm_fast_model = Task(name="lstm_fast_model",
                           create_model=create_model,
                           create_loss=create_loss,
                           create_optimizer=create_optimizer,
                           train_tf_record_path=train_tf_record_path,
                           valid_tf_record_path=valid_tf_record_path,
                           test_tf_record_path=test_tf_record_path,
                           hyperparams=hyperparams,
                           checkpoint_path=abs_save_path,
                           training_hooks=[LearningRateDecayHook, InitLegacyModelHook],
                           evaluation_hooks=[InitLegacyModelHook])

    # Example of training each model separately, they will share weight if the hidden layers
    # are the same and if their checkpoint path is the same
    # trainer.train_classic(task=pos_classifier)
    trainer.train_classic(task=lstm_fast_model)

    # Same as above, only more intuitive
    # trainer.train_transfer_learning(tasks=[pos_classifier, lstm_fast_model])
    #
    # Train simultaneously, you can decide to switch the training each epoch or each batch or both
    # trainer.train_multitask_learning(tasks=[pos_classifier, lstm_fast_model],
    #                                  switch_each_epoch=True,
    #                                  switch_each_batch=False,
    #                                  num_multitask_epochs=hyperparams.train.num_epochs)

    print("End training")
