from rnnlm.classes.predictor import Predictor
from rnnlm.classes.trainer import Trainer
from rnnlm.utils.estimator.estimator_hook.early_stopping import EarlyStoppingHook
from rnnlm.utils.estimator.estimator_hook.learning_rate_decay import LearningRateDecayHook
from datetime import datetime
import os
import importlib


class ExperimentsRunner:
    """
    Runs the desired experiments from the config file
    """

    def __init__(self, config):
        self.experiment_config = config

    def run(self):

        for experiment in self.experiment_config.experiments:

            # TODO - global var managing
            # define hooks global vars
            EarlyStoppingHook.should_stop = False
            LearningRateDecayHook.epoch_counter = 0

            print("{} running experiment {}".format(datetime.now(), experiment.name))

            # TODO Test and manage the logic so it will fit to train a single model
            shared_hyperparams = experiment.hyperparameters.shared_layer

            abs_save_path = os.path.join(os.getcwd(), shared_hyperparams.problem.save_path)

            shared_model_name = shared_hyperparams.create_model
            shared_layer_builder = importlib.import_module("rnnlm.models.{}.model".format(shared_model_name))

            if experiment.get_or_default(key="predict_mode", default=False):
                self._run_prediction_experiment(experiment=experiment,
                                                create_model=shared_layer_builder.create_model,
                                                shared_model_name=shared_model_name,
                                                shared_hyperparams=shared_hyperparams,
                                                checkpoint_path=abs_save_path)

            else:
                self._run_training_and_evaluation_experiment(experiment=experiment,
                                                             create_model=shared_layer_builder.create_model,
                                                             shared_hyperparams=shared_hyperparams,
                                                             checkpoint_path=abs_save_path)

            print("{} done running experiment {}".format(datetime.now(), experiment.name))

    def _run_prediction_experiment(self, experiment, create_model, shared_model_name, shared_hyperparams, checkpoint_path):
        create_model_hyperparams = experiment.hyperparameters.get_or_default(key=shared_model_name, default=None)

        if not create_model_hyperparams:
            raise ValueError("must defined the name of the model that will build the shared layer")
        predictor = Predictor(create_model=create_model,
                              checkpoint_path=checkpoint_path,
                              shared_hyperparams=shared_hyperparams,
                              create_model_hyperparams=create_model_hyperparams)
        predictor.predict()

    def _run_training_and_evaluation_experiment(self,
                                                experiment,
                                                create_model,
                                                shared_hyperparams,
                                                checkpoint_path):

        trainer = Trainer(create_model=create_model,
                          checkpoint_path=checkpoint_path,
                          shared_hyperparams=shared_hyperparams)

        tasks_hyperparams = dict()
        for model in experiment.models:
            tasks_hyperparams[model] = self._collect_model_hyperparams(model, experiment)

            pre_training = importlib.import_module("rnnlm.models.{}.pre_training".format(model))
            task = importlib.import_module("rnnlm.models.{}.task".format(model))

            if tasks_hyperparams[model].problem.convert_raw_to_tf_records:
                abs_tf_record_path = os.path.join(os.getcwd(), shared_hyperparams.problem.tf_records_path)
                if not os.path.exists(abs_tf_record_path):
                    os.makedirs(abs_tf_record_path)

                print("Converting raw data to tfrecord format")
                pre_training.main(shared_hyperparams=shared_hyperparams,
                                  hyperparams=tasks_hyperparams[model])

            task_to_train = task.create_task(shared_hyperparams=shared_hyperparams,
                                             hyperparams=tasks_hyperparams[model])
            trainer.add_task(task_to_train)

        learning_technique = experiment.learning_technique

        print("Start training")
        self._train_according_to_learning_technique(trainer, learning_technique, shared_hyperparams)
        print("End training")

    def _collect_model_hyperparams(self, model, experiment):
        model_hyperparams = experiment.hyperparameters.get_or_default(key=model, default=None)

        if not model_hyperparams:
            raise ValueError(
                "no hyperparams found for model {}. Make sure the name {} exists under hyperparams settings"
                    .format(model, model)
            )

        if not model_hyperparams.problem.data_path:
            raise ValueError("Must set data_path in hyperparams config for model {}".format(model))

        return model_hyperparams

    def _train_according_to_learning_technique(self, trainer, learning_technique, shared_hyperparams):
        if learning_technique == "normal":
            trainer.train_normal()
        elif learning_technique == "transfer":
            trainer.train_transfer_learning()
        elif learning_technique == "multitask":
            switch_each_epoch = shared_hyperparams.train.switch_each_epoch
            switch_each_batch = shared_hyperparams.train.switch_each_batch
            num_multitask_epochs = shared_hyperparams.train.num_multitask_epochs

            trainer.train_multitask(switch_each_epoch, switch_each_batch, num_multitask_epochs)
        else:
            raise ValueError(
                "unsupported learning technique {}\nonly normal, transfer and multitask are supported.".format(
                    learning_technique)
            )
