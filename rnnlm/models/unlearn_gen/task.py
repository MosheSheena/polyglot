from rnnlm.models.unlearn_gen.loss import create_loss
from rnnlm.models.unlearn_gen.optimizer import create_optimizer
from rnnlm.utils.estimator.estimator_hook.loss import ShowNormalLossHook
from rnnlm.utils.estimator.estimator_hook.learning_rate_decay import LearningRateDecayHook
from rnnlm.utils.task import Task


def create_task(hyperparams):

    lstm_fast_model = Task(name="generated_sentence_classifier",
                           create_loss=create_loss,
                           create_optimizer=create_optimizer,
                           hyperparams=hyperparams,
                           training_hooks=[ShowNormalLossHook, LearningRateDecayHook],
                           evaluation_hooks=[ShowNormalLossHook])
    return lstm_fast_model
