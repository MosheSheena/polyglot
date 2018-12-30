from rnnlm.classes.trainer import Trainer
from rnnlm.utils.hyperparams import load_params
from rnnlm.models.lstm_fast.model import create_model
import os
import importlib
from datetime import datetime


if __name__ == "__main__":
    print("start time: {}".format(datetime.now()))
    experiment_config = load_params(os.path.join(os.getcwd(), "experiment_config.json"))

    for experiment in experiment_config.experiments:
        print("{} running experiment {}".format(datetime.now(), experiment.name))
        shared_hyperparams = experiment.hyperparameters.shared_layer

        abs_save_path = os.path.join(os.getcwd(), shared_hyperparams.problem.save_path)
        trainer = Trainer(create_model=create_model,
                          checkpoint_path=abs_save_path,
                          shared_hyperparams=shared_hyperparams)

        tasks_hyperparams = dict()
        for model in experiment.models:
            tasks_hyperparams[model] = experiment.hyperparameters.get_or_default(key=model, default=None)

            # TODO - perform as many check as possible on hyperparams JSON
            if not tasks_hyperparams[model].problem.data_path:
                raise ValueError("Must set data_path hyperparameters.json")

            pre_training = importlib.import_module("rnnlm.models.{}.pre_training".format(model))
            task = importlib.import_module("rnnlm.models.{}.task".format(model))

            if tasks_hyperparams[model].problem.convert_raw_to_tf_records:
                print("Converting raw data to tfrecord format")
                pre_training.main(shared_hyperparams=shared_hyperparams,
                                  hyperparams=tasks_hyperparams[model])

            task_to_train = task.create_task(shared_hyperparams=shared_hyperparams,
                                             hyperparams=tasks_hyperparams[model])
            trainer.add_task(task_to_train)

        technique = experiment.learning_technique
        print("Start training")

        if technique == "normal":
            trainer.train_normal()
        elif technique == "transfer":
            trainer.train_transfer_learning()
        elif technique == "multitask":
            switch_each_epoch = shared_hyperparams.train.switch_each_epoch
            switch_each_batch = shared_hyperparams.train.switch_each_batch
            num_multitask_epochs = shared_hyperparams.train.num_multitask_epochs

            trainer.train_multitask(switch_each_epoch, switch_each_batch, num_multitask_epochs)
        else:
            raise NotImplemented(
                "unsupported learning technique {}\nonly normal, transfer and multitask are supported.".format(technique)
            )

        print("End training")
        print("{} done running experiment {}".format(datetime.now(), experiment.name))

    print("end time: {}".format(datetime.now()))
