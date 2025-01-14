import logging.config

import tensorflow as tf
import yaml

from config.log import config as rnnlm_config

logging.config.dictConfig(yaml.load(open(rnnlm_config.LOGGING_CONF_PATH, 'r')))
logger = logging.getLogger('rnnlm.utils.estimator.estimator_hook.learning_rate_decay')


class LearningRateDecayHook(tf.train.SessionRunHook):

    """
    Decays the learning rate according to num of epochs that were ran so far.
    This class implements the logic of decaying the learning rate like the legacy model did.
    """

    # each time we call an estimator method (train, eval or predict) an object from this class is instantiated
    # we need to declare a static var because we want to save the counter's state between instantiations of this class
    epoch_counter = 0

    def __init__(self, hyperparameters, optimizer_params, **kwargs):
        self.lr_update_op = optimizer_params["lr_update_op"]
        self.current_lr = optimizer_params["learning_rate"]
        self.new_lr = optimizer_params["new_lr_placeholder"]
        self.hyperparams = hyperparameters

    def after_create_session(self, session, coord):
        lr_decay = self.hyperparams.train.learning_rate.decay ** max(
            LearningRateDecayHook.epoch_counter + 1 - self.hyperparams.train.learning_rate.decay_max_factor, 0.0)
        assign_lr(session,
                  self.lr_update_op,
                  self.hyperparams.train.learning_rate.start_value * lr_decay,
                  self.new_lr)
        logger.info("learning rate=%s", session.run(self.current_lr))

    def end(self, session):
        LearningRateDecayHook.epoch_counter += 1


def assign_lr(session, lr_update, lr_value, new_lr):
    """
    Assigns a new learning rate
    Args:
        session: (tf.Session)
        lr_update: (Tensor) tf.assign op tensor
        lr_value: (int) the new value for the learning rate
        new_lr: (Placeholder) a placeholder for the learning rate
    Returns:
        None
    """
    session.run(lr_update, feed_dict={new_lr: lr_value})
