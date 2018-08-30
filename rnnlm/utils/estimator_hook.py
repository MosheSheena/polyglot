import tensorflow as tf


class EstimatorHook(tf.train.SessionRunHook):

    def __init__(self, model, losses, hyperparams, epoch_size, eval_op=None, verbose=False):
        self.losses = losses

    def before_run(self, run_context):
        self.losses["step"] = tf.train.get_global_step()
        run_args = tf.train.SessionRunArgs(fetches=self.losses)
        return run_args

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):

        results = run_values.results

        print("{}: Step {}, Loss {}".format(self.mode, self.report_storage[self.mode]["step"][-1],
                                            self.report_storage[self.mode]["loss"][-1]))

