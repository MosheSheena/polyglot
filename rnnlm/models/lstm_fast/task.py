from rnnlm.models.lstm_fast.loss import create_loss
from rnnlm.models.lstm_fast.optimizer import create_optimizer
from rnnlm.utils.estimator.estimator_hook.learning_rate_decay import LearningRateDecayHook
from rnnlm.utils.estimator.estimator_hook.init_legacy_model import InitLegacyModelHook
from rnnlm.utils.estimator.estimator_hook.perplexity import MeasurePerplexityHook
from rnnlm.utils.task import Task
import os


def create_task(shared_hyperparams, hyperparams):

    abs_tf_record_path = os.path.join(os.getcwd(), shared_hyperparams.problem.tf_records_path)

    train_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.problem.tf_record_train_file)
    valid_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.problem.tf_record_valid_file)
    test_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.problem.tf_record_test_file)

    lstm_fast_model = Task(name="lstm_fast_model",
                           create_loss=create_loss,
                           create_optimizer=create_optimizer,
                           train_tf_record_path=train_tf_record_path,
                           valid_tf_record_path=valid_tf_record_path,
                           test_tf_record_path=test_tf_record_path,
                           hyperparams=hyperparams,
                           training_hooks=[MeasurePerplexityHook, LearningRateDecayHook, InitLegacyModelHook],
                           evaluation_hooks=[MeasurePerplexityHook, InitLegacyModelHook])
    return lstm_fast_model
