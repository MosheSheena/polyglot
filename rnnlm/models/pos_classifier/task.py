from rnnlm.utils.task import Task
from rnnlm.models.pos_classifier.loss import create_loss
from rnnlm.models.pos_classifier.optimizer import create_optimizer
from rnnlm.utils.estimator.estimator_hook.loss import ShowNormalLossHook
from rnnlm.utils.estimator.estimator_hook.learning_rate_decay import LearningRateDecayHook


def create_task(hyperparams):
    pos_classifier = Task(name="pos_classifier",
                          create_loss=create_loss,
                          create_optimizer=create_optimizer,
                          hyperparams=hyperparams,
                          training_hooks=[ShowNormalLossHook, LearningRateDecayHook],
                          evaluation_hooks=[ShowNormalLossHook])
    return pos_classifier
