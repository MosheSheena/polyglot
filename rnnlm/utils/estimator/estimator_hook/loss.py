import logging.config

import tensorflow as tf
import yaml

from rnnlm import config as rnnlm_config

logging.config.dictConfig(yaml.load(open(rnnlm_config.LOGGING_CONF_PATH, 'r')))
logger = logging.getLogger('rnnlm.utils.estimator.estimator_hook.loss')


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
