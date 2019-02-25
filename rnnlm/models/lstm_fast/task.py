from rnnlm.models.lstm_fast.loss import create_loss
from rnnlm.models.lstm_fast.optimizer import create_optimizer
from rnnlm.utils.estimator.estimator_hook.learning_rate_decay import LearningRateDecayHook
from rnnlm.utils.estimator.estimator_hook.init_legacy_model import InitLegacyModelHook
from rnnlm.utils.estimator.estimator_hook.perplexity import MeasurePerplexityHook
from rnnlm.utils.task import Task


def create_task(hyperparams):

    lstm_fast_model = Task(name="lstm_fast_model",
                           create_loss=create_loss,
                           create_optimizer=create_optimizer,
                           hyperparams=hyperparams,
                           training_hooks=[MeasurePerplexityHook, LearningRateDecayHook, InitLegacyModelHook],
                           evaluation_hooks=[MeasurePerplexityHook, InitLegacyModelHook])
    return lstm_fast_model
