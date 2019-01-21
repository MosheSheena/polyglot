import os

from rnnlm.utils.tf_io.io_service import load_dataset, create_dataset_from_tensor
from rnnlm.utils.estimator.estimator_hook.early_stopping import EarlyStoppingHook
from collections import defaultdict
from shutil import copy2

import tensorflow as tf
import numpy as np

# like tf.train.global_step, only per dataset
dataset_step_counter = defaultdict(int)
PROJECTOR_METADATA_FILE_NAME = 'metadata.tsv'
PROJECTOR_CONFIG_FILE = 'config/projector/projector_config.pbtxt'


def _create_tf_estimator_spec(create_model,
                              create_loss,
                              create_optimizer,
                              shared_hyperparams,
                              training_hooks=None,
                              evaluation_hooks=None):
    """
    create a generic model_fn required for the estimator to train
    Args:
        create_model (func): function defining the model
        create_loss (func): function defining the loss
        create_optimizer (func): function defining the optimizer
        training_hooks ([<Class>]): list of classes that inheritance from tf.train.SessionRunHook,
         defining training hooks.
         second is a list of the args that that are required to create an instance of that hook.
        evaluation_hooks ([<Class>]): list of classes that inheritance from tf.train.SessionRunHook,
         defining evaluation hooks.

    Returns:
        (func) the model_fn required by tf.Estimator
    """

    def my_model_fn(features, labels, mode, params):
        # Talk to the outside world, this dict will be pass to any hooks that
        # are created outside and passed here.
        estimator_params = dict()
        estimator_params["hyperparameters"] = params
        estimator_params["shared_hyperparameters"] = shared_hyperparams
        estimator_params["mode"] = mode

        # Create a model
        model = create_model(features, mode, params, shared_hyperparams)
        estimator_params["model"] = model

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "logits": model["logits"],
                "arg_max": tf.argmax(model["logits"], 1)
                #"softmax": tf.nn.softmax(model["logits"])
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Create a loss
        loss, metrics = create_loss(model, labels, params)
        estimator_params["loss"] = loss
        estimator_params["metrics"] = metrics

        _training_hooks = list()
        _evaluation_hooks = list()

        if mode == tf.estimator.ModeKeys.EVAL:

            if evaluation_hooks is not None:
                for hook_class in evaluation_hooks:
                    hook_instance = hook_class(**estimator_params)
                    _evaluation_hooks.append(hook_instance)

            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              eval_metric_ops=metrics,
                                              evaluation_hooks=_evaluation_hooks)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Create an optimizer
            train_op, optimizer_params = create_optimizer(loss, params)
            estimator_params["train_op"] = train_op
            estimator_params["optimizer_params"] = optimizer_params

            if training_hooks is not None:
                for hook_class in training_hooks:
                    hook_instance = hook_class(**estimator_params)
                    _training_hooks.append(hook_instance)

            stop_params = shared_hyperparams.train.early_stopping
            early_stop = EarlyStoppingHook(loss_tensor=loss,
                                           threshold=stop_params.threshold,
                                           max_steps_without_improvement=stop_params.max_steps_without_improvement)
            _training_hooks.append(early_stop)

            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks=_training_hooks)

        raise RuntimeError(
            "Unexpected mode. mode can be {} or {} or {} but got {}".format(
                tf.estimator.ModeKeys.PREDICT,
                tf.estimator.ModeKeys.EVAL,
                tf.estimator.ModeKeys.TRAIN,
                mode
            )
        )

    return my_model_fn


def _create_input_fn(tf_record_path, hyperparams, shared_hyperparams):
    def input_fn():
        """
        This method is invoke each time we call estimator.train
        or estimator.eval or estimator.predict, the estimator recreates
        the tf.data.Dataset that we return here. For each dataset we train,
        we want to remember how many records we trained it so we can iterate
        different datasets each epoch or batch. Therefore, we create a dataset
        step counter for each new dataset that we load, and skip records from
        reloaded dataset according to it's step.
        Returns:
            tf.data.Dataset object representing the dataset
        """
        global dataset_step_counter
        # TODO replace with logging
        print("path = {}".format(tf_record_path))
        print("dataset_steps = {}".format(dataset_step_counter[tf_record_path]))
        shuffle = hyperparams.train.get_or_default(key="shuffle", default=False)
        shuffle_buffer_size = hyperparams.train.get_or_default(key="shuffle_buffer_size", default=10000)
        dataset = load_dataset(abs_tf_record_path=tf_record_path,
                               batch_size=hyperparams.train.batch_size,
                               seq_len=shared_hyperparams.arch.sequence_length,
                               skip_first_n=dataset_step_counter[tf_record_path],
                               shuffle=shuffle,
                               shuffle_buffer_size=shuffle_buffer_size)
        # s = get_epoch_size_from_tf_dataset(dataset)
        # print(s)
        return dataset

    return input_fn


def _train_estimator(estimator, dataset, tf_record_path, steps):
    estimator.train(input_fn=dataset, steps=steps)
    dataset_step_counter[tf_record_path] += steps


def _evaluate_estimator(estimator, dataset, tf_record_path, steps):
    estimator.evaluate(input_fn=dataset, steps=steps)
    dataset_step_counter[tf_record_path] += steps


def create_prediction_estimator(create_model, prediction_dict, checkpoint_path, shared_hyperparams, hyperparams):

    estimator_spec = _create_tf_estimator_spec(create_model=create_model,
                                               create_loss=None,
                                               create_optimizer=None,
                                               shared_hyperparams=shared_hyperparams)

    estimator = tf.estimator.Estimator(estimator_spec,
                                       model_dir=checkpoint_path,
                                       params=hyperparams)

    predict_input = create_dataset_from_tensor(tensor=prediction_dict,
                                               batch_size=hyperparams.train.batch_size)

    return estimator, predict_input


def train_and_evaluate_model(create_model,
                             create_loss,
                             create_optimizer,
                             train_tf_record_path,
                             valid_tf_record_path,
                             test_tf_record_path,
                             num_epochs,
                             epoch_size_train,
                             epoch_size_valid,
                             epoch_size_test,
                             hyperparams,
                             shared_hyperparams,
                             checkpoint_path=None,
                             training_hooks=None,
                             evaluation_hooks=None):
    """
    Invoke tf.estimator with the passed args

    Args:
        create_model (func): creates the model, the function arguments must be (features, mode, params)
            where feature is the input_fn (in our case the input pipeline from tf.data) mode is an instance of
            tf.estimator.ModeKeys and params is a dict containing hyperparams used in model
        create_loss (func): defines the loss, receives as args the model in a dict from create model,
            the labels and hyperparams. Must return a dict containing key 'cost' the is the loss as a scalar
        create_optimizer (func): defines the optimizer, receives as args the loss dict from create loss and hyperparams.
            Returns the train_op
        hyperparams (Dict2Obj): contains the hyperparams configuration
        shared_hyperparams (Dict2Obj): contains hyperparams that are shared between tasks
        train_tf_record_path (str): full path of train data in tf record format
        valid_tf_record_path (str): full path of valid data in tf record format
        test_tf_record_path (str): full path of test data in tf record format
        num_epochs (int): num of epochs to train
        epoch_size_train (int): how much iterations to extract all data from dataset
        epoch_size_valid (int): how much iterations to extract all data from dataset
        epoch_size_test (int): how much iterations to extract all data from dataset
        checkpoint_path (str): absolute path for model checkpoints
        training_hooks ([<Class>]): list of classes that inheritance from tf.train.SessionRunHook,
         defining training hooks.
         second is a list of the args that that are required to create an instance of that hook.
        evaluation_hooks ([<Class>]): list of classes that inheritance from tf.train.SessionRunHook,
         defining evaluation hooks.

    Returns:
        None
    """
    x = lambda: create_dataset_from_tensor(tensor=np.random.random_integers(0, 10000, (64, 20)),
                                           batch_size=hyperparams.train.batch_size)
    # Create labels for the embeddings projector
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    metadata_file = os.path.join(os.getcwd(), hyperparams.problem.vocab_path)
    destination = os.path.join(checkpoint_path, PROJECTOR_METADATA_FILE_NAME)
    copy2(metadata_file, destination)

    copy2(PROJECTOR_CONFIG_FILE, checkpoint_path)

    # Create the datasets
    train_dataset = _create_input_fn(tf_record_path=train_tf_record_path,
                                     hyperparams=hyperparams,
                                     shared_hyperparams=shared_hyperparams)
    validation_dataset = _create_input_fn(tf_record_path=valid_tf_record_path,
                                          hyperparams=hyperparams,
                                          shared_hyperparams=shared_hyperparams)
    test_dataset = _create_input_fn(tf_record_path=test_tf_record_path,
                                    hyperparams=hyperparams,
                                    shared_hyperparams=shared_hyperparams)

    # Create estimator spec object
    estimator_spec = _create_tf_estimator_spec(create_model=create_model,
                                               create_loss=create_loss,
                                               create_optimizer=create_optimizer,
                                               shared_hyperparams=shared_hyperparams,
                                               training_hooks=training_hooks,
                                               evaluation_hooks=evaluation_hooks)

    # Create estimator run config
    summary_steps = shared_hyperparams.train.get_or_default(key="summary_steps", default=100)
    save_checkpoint_steps = shared_hyperparams.train.get_or_default(key="save_checkpoint_steps", default=200)
    keep_checkpoints_max = shared_hyperparams.train.get_or_default(key="keep_checkpoint_max", default=5)
    config = tf.estimator.RunConfig(model_dir=checkpoint_path,
                                    save_summary_steps=summary_steps,
                                    save_checkpoints_steps=save_checkpoint_steps,
                                    keep_checkpoint_max=keep_checkpoints_max)
    # Create the estimator itself
    estimator = tf.estimator.Estimator(model_fn=estimator_spec,
                                       config=config,
                                       params=hyperparams)

    p = estimator.predict(x)
    next(p)
    for i in range(num_epochs):
        print("Starting training epoch #{}".format(i + 1))
        # Train and evaluate
        _train_estimator(estimator=estimator,
                         dataset=train_dataset,
                         tf_record_path=train_tf_record_path,
                         steps=epoch_size_train)
        _evaluate_estimator(estimator=estimator,
                            dataset=validation_dataset,
                            tf_record_path=valid_tf_record_path,
                            steps=epoch_size_valid)
        print("Finished training epoch #{}".format(i + 1))
        if EarlyStoppingHook.should_stop:
            print("early stopping detected, stop training")
            break

    _evaluate_estimator(estimator=estimator,
                        dataset=test_dataset,
                        tf_record_path=test_tf_record_path,
                        steps=epoch_size_test)
