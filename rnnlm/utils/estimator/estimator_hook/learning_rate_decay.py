import tensorflow as tf

from rnnlm.utils.estimator.estimator_hook import assign_lr


class LearningRateDecayHook(tf.train.SessionRunHook):

    """
    Decays the learning rate according to num of epochs that were ran so far.
    This class implements the logic of decaying the learning rate like the legacy model did.
    """

    # each time we call an estimator method (train, eval or predict) an object from this class is instantiated
    # we need to declare a static var because we want to save the counter's state between instantiations of this class
    epoch_counter = 0

    def __init__(self, estimator_params):
        self.lr_update_op = estimator_params["optimizer_params"]["lr_update_op"]
        self.current_lr = estimator_params["optimizer_params"]["learning_rate"]
        self.new_lr = estimator_params["optimizer_params"]["new_lr_placeholder"]
        self.hyperparams = estimator_params["hyperparameters"]

    def after_create_session(self, session, coord):
        lr_decay = self.hyperparams.train.learning_rate.decay ** max(
            LearningRateDecayHook.epoch_counter + 1 - self.hyperparams.train.learning_rate.decay_max_factor, 0.0)
        assign_lr(session,
                  self.lr_update_op,
                  self.hyperparams.train.learning_rate.start_value * lr_decay,
                  self.new_lr)
        print("Learning rate: {}".format(session.run(self.current_lr)))

    def end(self, session):
        LearningRateDecayHook.epoch_counter += 1
        print("Done {} epochs".format(LearningRateDecayHook.epoch_counter))


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