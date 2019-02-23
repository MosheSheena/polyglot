import os
from rnnlm.utils.task import Task
from rnnlm.models.pos_classifier.loss import create_loss
from rnnlm.models.pos_classifier.optimizer import create_optimizer
from rnnlm.utils.estimator.estimator_hook.loss import ShowNormalLossHook
from rnnlm.utils.estimator.estimator_hook.learning_rate_decay import LearningRateDecayHook


def create_task(hyperparams):
    abs_tf_record_path = os.path.join(os.getcwd(), hyperparams.data.tf_records_path)

    train_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.data.tf_record_train_file)
    valid_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.data.tf_record_valid_file)
    test_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.data.tf_record_test_file)

    pos_classifier = Task(name="pos_classifier",
                          create_loss=create_loss,
                          create_optimizer=create_optimizer,
                          train_tf_record_path=train_tf_record_path,
                          valid_tf_record_path=valid_tf_record_path,
                          test_tf_record_path=test_tf_record_path,
                          hyperparams=hyperparams,
                          training_hooks=[ShowNormalLossHook, LearningRateDecayHook],
                          evaluation_hooks=[ShowNormalLossHook])
    return pos_classifier
