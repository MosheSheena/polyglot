from rnnlm.models.gen_classifier.loss import create_loss
from rnnlm.models.gen_classifier.optimizer import create_optimizer
from rnnlm.utils.estimator.estimator_hook.loss import ShowNormalLossHook
from rnnlm.utils.estimator.estimator_hook.learning_rate_decay import LearningRateDecayHook
from rnnlm.utils.task import Task
import os


def create_task(hyperparams):

    abs_tf_record_path = os.path.join(os.getcwd(), hyperparams.data.tf_records_path)

    train_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.data.tf_record_train_file)
    valid_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.data.tf_record_valid_file)
    test_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.data.tf_record_test_file)

    lstm_fast_model = Task(name="generated_sentence_classifier",
                           create_loss=create_loss,
                           create_optimizer=create_optimizer,
                           train_tf_record_path=train_tf_record_path,
                           valid_tf_record_path=valid_tf_record_path,
                           test_tf_record_path=test_tf_record_path,
                           hyperparams=hyperparams,
                           training_hooks=[ShowNormalLossHook, LearningRateDecayHook],
                           evaluation_hooks=[ShowNormalLossHook])
    return lstm_fast_model
