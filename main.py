import logging.config
import os

import yaml

from rnnlm import config as rnnlm_config
from rnnlm.utils.experiment.experiments_runner import ExperimentsRunner
from rnnlm.utils.tf_io.hyperparams.hyperparams import load_params

logging.config.dictConfig(yaml.load(open(rnnlm_config.LOGGING_CONF_PATH, 'r')))
logger = logging.getLogger('main')

if __name__ == "__main__":
    logger.info("starting experiment execution")

    config_file = os.path.join(os.getcwd(), "experiment_config.json")
    config = load_params(config_file)

    experiment_runner = ExperimentsRunner(config)
    experiment_runner.run()

    logger.info("finished experiment execution")
