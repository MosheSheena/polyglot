import tensorflow as tf
import numpy as np
import time


class EstimatorHook(tf.train.SessionRunHook):

    def __init__(self, model, losses, hyperparams, epoch_size, eval_op=None, verbose=False):
        self.model = model
        self.losses = losses
        self.hyperparams = hyperparams
        self.epoch_size = epoch_size
        self.eval_op = eval_op
        self.verbose = verbose
        self.start_time = time.time()
        self.costs = 0.0
        self.iterations = 0
        self.step = 0

    def before_run(self, run_context):
        self.losses["step"] = tf.train.get_global_step()
        fetches = {
            "initial_state": self.model["initial_state"],
            "cost": self.losses["cost"],
            "final_state": self.model["final_state"]
        }

        if self.eval_op is not None:
            fetches["eval_op"] = self.eval_op

        feed_dict = {}
        # for step in range(self.epoch_size):
        #     for i, (c, h) in enumerate(self.model["initial_state"]):
        #         feed_dict[c] = state[i].c
        #         feed_dict[h] = state[i].h

        run_args = tf.train.SessionRunArgs(fetches=self.losses, feed_dict=feed_dict)
        return run_args

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):

        results = run_values.results

        cost = results["cost"]
        state = results["final_state"]

        self.costs += cost
        self.iterations += self.hyperparams.arch.hidden_layer_depth

        if self.verbose and self.step % (self.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (self.step * 1.0 / self.epoch_size, np.exp(self.costs / self.iterations),
                   self.iterations * self.hyperparams.train.batch_size / (time.time() - self.start_time)))

        self.step += 1
