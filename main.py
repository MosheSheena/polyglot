import os
from datetime import datetime
from rnnlm.utils.experiment.experiments_runner import ExperimentsRunner
from rnnlm.utils.tf_io.hyperparams.hyperparams import load_params


if __name__ == "__main__":
    print("start time: {}".format(datetime.now()))

    config_file = os.path.join(os.getcwd(), "experiment_config.json")
    config = load_params(config_file)

    experiment_runner = ExperimentsRunner(config)
    experiment_runner.run()

    print("end time: {}".format(datetime.now()))
