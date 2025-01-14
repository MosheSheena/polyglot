import logging.config
import time

import numpy as np
import tensorflow as tf
import yaml

from config.log import config as rnnlm_config

logging.config.dictConfig(yaml.load(open(rnnlm_config.LOGGING_CONF_PATH, 'r')))
logger = logging.getLogger('rnnlm.utils.estimator.estimator_hook.perplexity')


class MeasurePerplexityHook(tf.train.SessionRunHook):
    """
    Measure perplexity of the current language model
    """

    def __init__(self, loss, mode, hyperparameters, shared_hyperparameters, **kwargs):
        self.loss = loss
        self.hyperparams = hyperparameters
        self.shared_hyperparams = shared_hyperparameters
        self.start_time = time.time()
        self.costs = 0.0
        self.iterations = 0
        self.step = 0
        self.mode = mode

    def before_run(self, run_context):
        fetches = {"cost": self.loss}

        run_args = tf.train.SessionRunArgs(fetches=fetches)
        return run_args

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):

        results = run_values.results

        cost = results["cost"]

        self.costs += cost
        self.iterations += self.shared_hyperparams.arch.sequence_length

        if self.step % 100 == 0:
            ppl = np.exp(self.costs / self.iterations)
            wps = self.iterations * self.hyperparams.train.batch_size / (time.time() - self.start_time)
            logger.info(
                "mode: %s perplexity: %.3f speed: %.0f wps",
                self.mode,
                ppl,
                wps
            )

        self.step += 1

    def end(self, session):
        logger.info("mode: %s perplexity: %.3f", self.mode, np.exp(self.costs / self.iterations))
