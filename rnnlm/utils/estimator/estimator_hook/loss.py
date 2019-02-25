import logging.config

import tensorflow as tf
import yaml

from rnnlm import config as rnnlm_config

logging.config.dictConfig(yaml.load(open(rnnlm_config.LOGGING_CONF_PATH, 'r')))
logger = logging.getLogger('rnnlm.utils.estimator.estimator_hook.loss')


class ShowNormalLossHook(tf.train.SessionRunHook):

    def __init__(self, loss, mode, **kwargs):
        self.loss = loss
        self.mode = mode
        self.steps = 0
        self.current_loss = 0

    def before_run(self, run_context):
        fetches = {"loss": self.loss}

        run_args = tf.train.SessionRunArgs(fetches=fetches)
        return run_args

    def after_run(self, run_context, run_values):
        results = run_values.results

        self.current_loss = results["loss"]
        if self.steps % 100 == 0:
            logger.info("mode: {}, loss: {}".format(self.mode, self.current_loss))
        self.steps += 1

    def end(self, session):
        logger.info("mode: {}, loss: {}".format(self.mode, self.current_loss))
