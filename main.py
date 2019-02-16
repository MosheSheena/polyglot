import logging
import os

from rnnlm.utils.experiment.experiments_runner import ExperimentsRunner
from rnnlm.utils.tf_io.hyperparams.hyperparams import load_params

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_formatter = formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('main.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(file_formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
ch.setFormatter(console_formatter)

logger.addHandler(fh)
logger.addHandler(ch)


if __name__ == "__main__":
    logger.info("starting experiment execution")

    config_file = os.path.join(os.getcwd(), "experiment_config.json")
    config = load_params(config_file)

    experiment_runner = ExperimentsRunner(config)
    experiment_runner.run()

    logger.info("finished experiment execution")
