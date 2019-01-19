import os
from datetime import datetime
from rnnlm.classes.experiment_runner import ExperimentRunner


if __name__ == "__main__":
    print("start time: {}".format(datetime.now()))

    config = os.path.join(os.getcwd(), "experiment_config.json")
    experiment_runner = ExperimentRunner(config)
    experiment_runner.run()

    print("end time: {}".format(datetime.now()))
