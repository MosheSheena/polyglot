import tensorflow as tf


class ShowNormalLossHook(tf.train.SessionRunHook):

    def __init__(self, estimator_params) -> None:
        self.loss = estimator_params["loss"]
        self.steps = 0

    def before_run(self, run_context):
        fetches = {"loss": self.loss}

        run_args = tf.train.SessionRunArgs(fetches=fetches)
        return run_args

    def after_run(self, run_context, run_values):
        results = run_values.results

        current_loss = results["loss"]
        if self.steps % 100 == 0:
            print("loss: {}".format(current_loss))
        self.steps += 1
