import tensorflow as tf


class EstimatorHook(tf.train.SessionRunHook):

    def __init__(self, model, losses, hyperparams, epoch_size, eval_op=None, verbose=False):
        self.model = model
        self.losses = losses
        self.hyperparams = hyperparams
        self.epoch_size = epoch_size
        self.eval_op = eval_op
        self.verbose = verbose

    def before_run(self, run_context):
        self.losses["step"] = tf.train.get_global_step()
        fetches = {
            "initial_state": self.model["initial_state"],
            "cost": self.losses["cost"],
            "final_state": self.model["final_state"]
        }

        if self.eval_op is not None:
            fetches["eval_op"] = self.eval_op

        for step in range(self.epoch_size):
            feed_dict = {}
            for i, (c, h) in enumerate(self.model["initial_state"]):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

        run_args = tf.train.SessionRunArgs(fetches=self.losses)
        return run_args

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):

        results = run_values.results

        print("{}: Step {}, Loss {}".format(self.mode, self.report_storage[self.mode]["step"][-1],
                                            self.report_storage[self.mode]["loss"][-1]))

