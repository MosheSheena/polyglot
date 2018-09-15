import tensorflow as tf
import numpy as np
import time


class EstimatorHook(tf.train.SessionRunHook):
    """
    This class allows us to integrate with the tf.estimator class in run time.
    the method before_run gets called inside tf.estimator before performing sess.run
    the method after_run gets called inside tf.estimator after performing sess.run

    this class contains the logic of the run_epoch method from the legacy model
    """

    def __init__(self, model, losses, hyperparams, verbose=False):
        self.model = model
        self.losses = losses
        self.hyperparams = hyperparams
        self.verbose = verbose
        self.start_time = time.time()
        self.costs = 0.0
        self.iterations = 0
        self.step = 0

        # set the state to the initial state 
        self.state = self.model["initial_state"]

    def before_run(self, run_context):
        self.losses["step"] = tf.train.get_global_step()
        fetches = {
            "initial_state": self.model["initial_state"],
            "cost": self.losses["cost"],
            "final_state": self.model["final_state"]
        }

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

        if self.verbose and self.step % 10000 == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (self.step * 1.0 / self.epoch_size, np.exp(self.costs / self.iterations),
                   self.iterations * self.hyperparams.train.batch_size / (time.time() - self.start_time)))

        self.step += 1
