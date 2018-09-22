import tensorflow as tf
import numpy as np
import time


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


class EstimatorHook(tf.train.SessionRunHook):
    """
    This class allows us to integrate with the tf.estimator class in run time.
    the method before_run gets called inside tf.estimator before performing sess.run
    the method after_run gets called inside tf.estimator after performing sess.run

    this class contains the logic of the run_epoch method from the legacy model
    """

    def __init__(self, model, losses, hyperparams, mode, verbose=True):
        self.model = model
        self.losses = losses
        self.hyperparams = hyperparams
        self.verbose = verbose
        self.start_time = time.time()
        self.costs = 0.0
        self.iterations = 0
        self.step = 0
        self.mode = mode
        self.state = None
        self.epoch_counter = 0

    def after_create_session(self, session, coord):
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            lr_decay = self.hyperparams.train.learning_rate.decay ** max(
                self.epoch_counter + 1 - self.hyperparams.train.learning_rate.decay_max_factor, 0.0)
            assign_lr(session,
                      self.losses["lr_update_op"],
                      self.hyperparams.train.learning_rate.start_value * lr_decay,
                      self.losses["new_lr"])
            print("Learning rate: {}".format(session.run(self.losses["lr"])))

        # set the state to the initial state
        self.state = session.run(self.model["initial_state"])

    def before_run(self, run_context):
        self.losses["step"] = tf.train.get_global_step()
        fetches = {
            "cost": self.losses["cost"],
            "final_state": self.model["final_state"]
        }

        # if self.mode == tf.estimator.ModeKeys.TRAIN:
        #     fetches["train_op"] = self.losses["train_op"]

        feed_dict = {}

        # update the feed_dict for the current state
        for i, (c, h) in enumerate(self.model["initial_state"]):
            feed_dict[c] = self.state[i].c
            feed_dict[h] = self.state[i].h

        run_args = tf.train.SessionRunArgs(fetches=fetches, feed_dict=feed_dict)
        return run_args

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):

        results = run_values.results

        cost = results["cost"]

        # update the state for the next run
        self.state = results["final_state"]

        self.costs += cost
        self.iterations += self.hyperparams.arch.hidden_layer_depth

        if self.verbose and self.step % 100 == 0:
            print("perplexity: %.3f speed: %.0f wps" %
                  (np.exp(self.costs / self.iterations),
                   self.iterations * self.hyperparams.train.batch_size / (time.time() - self.start_time)))

        self.step += 1

    def end(self, session):
        self.epoch_counter += 1
        print("Done {} epochs".format(self.epoch_counter))
