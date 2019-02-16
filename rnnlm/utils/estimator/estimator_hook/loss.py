import logging

import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_formatter = formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

fh = logging.FileHandler('trainer.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(file_formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
ch.setFormatter(console_formatter)

logger.addHandler(fh)
logger.addHandler(ch)


class ShowNormalLossHook(tf.train.SessionRunHook):

    def __init__(self, loss, **kwargs):
        self.loss = loss
        self.steps = 0

    def before_run(self, run_context):
        fetches = {"loss": self.loss}

        run_args = tf.train.SessionRunArgs(fetches=fetches)
        return run_args

    def after_run(self, run_context, run_values):
        results = run_values.results

        current_loss = results["loss"]
        if self.steps % 100 == 0:
            logger.debug("loss=%s", current_loss)
        self.steps += 1
