import importlib
import logging.config
import os

import yaml

from rnnlm import config as rnnlm_config

from rnnlm.utils.estimator.estimator_hook.early_stopping import EarlyStoppingHook
from rnnlm.utils.estimator.estimator_hook.learning_rate_decay import LearningRateDecayHook
from rnnlm.utils.predict.predictor import Predictor
from rnnlm.utils.task_data import TaskData
from rnnlm.utils.trainer import Trainer

logging.config.dictConfig(yaml.load(open(rnnlm_config.LOGGING_CONF_PATH, 'r')))
logger = logging.getLogger('rnnlm.utils.experiment.experiments_runner')


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

            logger.info("running experiment %s", experiment.name)

            shared_hyperparams = experiment.hyperparameters.shared_params

            abs_save_path = os.path.join(os.getcwd(), shared_hyperparams.data.save_path)
            experiment_results_path = os.path.join(abs_save_path, experiment.name)

            shared_model_name = shared_hyperparams.create_model
            shared_layer_builder = importlib.import_module("rnnlm.models.{}.model".format(shared_model_name))

            if experiment.get_or_default(key="predict_mode", default=False):
                self._run_prediction_experiment(experiment=experiment,
                                                create_model=shared_layer_builder.create_model,
                                                shared_model_name=shared_model_name,
                                                shared_hyperparams=shared_hyperparams,
                                                checkpoint_path=experiment_results_path)

            else:
                self._run_training_and_evaluation_experiment(experiment=experiment,
                                                             create_model=shared_layer_builder.create_model,
                                                             shared_hyperparams=shared_hyperparams,
                                                             checkpoint_path=experiment_results_path)

            logger.info("finished running experiment %s", experiment.name)

    def _run_prediction_experiment(self,
                                   experiment,
                                   create_model,
                                   shared_model_name,
                                   shared_hyperparams,
                                   checkpoint_path):
        create_model_hyperparams = experiment.hyperparameters.get_or_default(key=shared_model_name, default=None)

        if not create_model_hyperparams:
            raise ValueError("must define the name of the model that will build the shared layers")
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

        for model in experiment.models:
            tasks_hyperparams = self._collect_task_hyperparams(model, experiment)
            pre_training = importlib.import_module("rnnlm.models.{}.pre_training".format(model))
            task = importlib.import_module("rnnlm.models.{}.task".format(model))

            raw_files, tf_record_outputs, vocabs = _get_data_paths_each_task(tasks_hyperparams)
            if tasks_hyperparams.data.pre_train:
                features_vocab, labels_vocab = vocabs

                logger.info("converting raw data to tfrecord format in experiment %s", experiment.name)

                pre_training.main(raw_files=raw_files,
                                  tf_record_outputs=tf_record_outputs,
                                  features_vocab=features_vocab,
                                  labels_vocab=labels_vocab,
                                  shared_hyperparams=shared_hyperparams,
                                  hyperparams=tasks_hyperparams)

            task_to_train = task.create_task(hyperparams=tasks_hyperparams)
            train_tf_record, valid_tf_record, test_tf_record = tf_record_outputs
            task_data = TaskData(task=task_to_train,
                                 train_tf_record_path=train_tf_record,
                                 valid_tf_record_path=valid_tf_record,
                                 test_tf_record_path=test_tf_record)

            trainer.add_task(task_data)

        learning_technique = experiment.learning_technique

        logger.info("start training experiment %s", experiment.name)
        self._train_according_to_learning_technique(trainer, learning_technique, shared_hyperparams)
        logger.info("finished training experiment %s", experiment.name)

    def _collect_task_hyperparams(self, task, experiment):
        model_hyperparams = experiment.hyperparameters.get_or_default(key=task, default=None)

        if not model_hyperparams:
            raise ValueError(
                "no hyperparams found for task {}. Make sure the name {} exists under hyperparams settings"
                    .format(task, task)
            )

        if not model_hyperparams.data.data_path:
            raise ValueError("Must set data_path in hyperparams config for task {}".format(task))

        return model_hyperparams

    def _train_according_to_learning_technique(self, trainer, learning_technique, shared_hyperparams):
        if learning_technique == "normal":
            trainer.train_normal()
        elif learning_technique == "transfer":
            trainer.train_transfer_learning()
        elif learning_technique == "multitask":
            switch_each_epoch = shared_hyperparams.train.multitask.switch_each_epoch
            switch_each_batch = shared_hyperparams.train.multitask.switch_each_batch
            num_multitask_epochs = shared_hyperparams.train.multitask.num_multitask_epochs

            trainer.train_multitask(switch_each_epoch, switch_each_batch, num_multitask_epochs)
        else:
            raise ValueError(
                "unsupported learning technique {}\nonly normal, transfer and multitask are supported.".format(
                    learning_technique)
            )


def _get_data_paths_each_task(hyperparams):

    abs_data_path = os.path.join(os.getcwd(), hyperparams.data.data_path)
    abs_tf_record_path = os.path.join(abs_data_path, hyperparams.data.tf_records_path)

    if not os.path.exists(abs_tf_record_path):
        os.makedirs(abs_tf_record_path)

    abs_vocab_features_path = os.path.join(abs_data_path, hyperparams.data.vocab_path_features)
    abs_vocab_labels_path = os.path.join(abs_data_path, hyperparams.data.vocab_path_labels)

    train_raw_data_path = os.path.join(abs_data_path, hyperparams.data.train_raw_data_file)
    valid_raw_data_path = os.path.join(abs_data_path, hyperparams.data.valid_raw_data_file)
    test_raw_data_path = os.path.join(abs_data_path, hyperparams.data.test_raw_data_file)

    train_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.data.tf_record_train_file)
    valid_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.data.tf_record_valid_file)
    test_tf_record_path = os.path.join(abs_tf_record_path, hyperparams.data.tf_record_test_file)

    raw_files = (train_raw_data_path, valid_raw_data_path, test_raw_data_path)
    tf_record_outputs = (train_tf_record_path, valid_tf_record_path, test_tf_record_path)
    vocabs = (abs_vocab_features_path, abs_vocab_labels_path)

    return raw_files, tf_record_outputs, vocabs
