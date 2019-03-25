import logging.config
import os
from collections import defaultdict
from shutil import copy2

import numpy as np
import tensorflow as tf
import yaml

from config.log import config as rnnlm_config
from rnnlm.utils.estimator.estimator_hook.early_stopping import EarlyStoppingHook
from rnnlm.utils.tf_io.io_service import load_dataset, create_dataset_from_tensor, create_vocab

logging.config.dictConfig(yaml.load(open(rnnlm_config.LOGGING_CONF_PATH, 'r')))
logger = logging.getLogger('rnnlm.utils.estimator.estimator')

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
    create a generic model_fn required for the estimator to train, eval or predict
    Args:
        create_model (func): function defining the model
        create_loss (func): function defining the loss
        create_optimizer (func): function defining the optimizer
        training_hooks ([<Class>]): list of classes that inheritance from tf.train.SessionRunHook,
         defining training hooks.
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
        with tf.name_scope("Train"):
            model = create_model(features, mode, params, shared_hyperparams)
        estimator_params["model"] = model

        if mode == tf.estimator.ModeKeys.PREDICT:
            with tf.name_scope("predict"):
                softmax = tf.nn.softmax(model["logits"])
                predictions = {
                    "logits": model["logits"],
                    "arg_max": tf.argmax(softmax, 1),
                    "softmax": softmax
                }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Create a loss
        with tf.name_scope("loss"):
            loss, loss_dict = create_loss(model, labels, params)
        estimator_params["loss"] = loss
        estimator_params["metrics"] = loss_dict

        _training_hooks = list()
        _evaluation_hooks = list()

        if mode == tf.estimator.ModeKeys.EVAL:

            if evaluation_hooks is not None:
                for hook_class in evaluation_hooks:
                    hook_instance = hook_class(**estimator_params)
                    _evaluation_hooks.append(hook_instance)

            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              evaluation_hooks=_evaluation_hooks)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Create an optimizer
            with tf.name_scope("optimizer"):
                train_op, optimizer_params = create_optimizer(loss, loss_dict, params)
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


def _create_input_fn(tf_record_path, hyperparams):
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

        logger.debug("TF records path=%s", tf_record_path)
        logger.debug("dataset_steps=%s", dataset_step_counter[tf_record_path])

        logger.info("training with dataset=%s", tf_record_path)

        shuffle = hyperparams.train.get_or_default(key="shuffle", default=False)
        shuffle_buffer_size = hyperparams.train.get_or_default(key="shuffle_buffer_size", default=10000)
        dataset = load_dataset(abs_tf_record_path=tf_record_path,
                               batch_size=hyperparams.train.batch_size,
                               feature_sample_size=hyperparams.data.shape_size_features,
                               label_sample_size=hyperparams.data.shape_size_labels,
                               skip_first_n=dataset_step_counter[tf_record_path],
                               shuffle=shuffle,
                               shuffle_buffer_size=shuffle_buffer_size)
        return dataset

    return input_fn


def _train_estimator(estimator, dataset, tf_record_path, steps):
    estimator.train(input_fn=dataset, steps=steps)
    dataset_step_counter[tf_record_path] += steps


def _evaluate_estimator(estimator, dataset, tf_record_path, steps):
    estimator.evaluate(input_fn=dataset, steps=steps)
    dataset_step_counter[tf_record_path] += steps


def _predict_estimator(estimator, shared_hyperparams, hyperparams):
    vocab_size = hyperparams.data.vocab_size_features
    batch_size = hyperparams.train.batch_size
    seq_len = shared_hyperparams.arch.sequence_length

    def predict_data():
        return create_dataset_from_tensor(tensor=np.random.random_integers(0, vocab_size - 1, (batch_size, seq_len)),
                                          batch_size=hyperparams.train.batch_size)

    data_path = os.path.join(os.getcwd(), hyperparams.data.data_path)
    word_2_id = os.path.join(data_path, hyperparams.data.vocab_path_features)
    with open(word_2_id) as f:
        word_2_id = create_vocab(f)
    id_2_word = dict(zip(word_2_id.values(), word_2_id.keys()))

    predictions = estimator.predict(predict_data)

    num_predicted = 0
    # predict returns a generator object
    for p in predictions:
        predicted_word = id_2_word[p["arg_max"]]
        logger.info("predicted word=%s", predicted_word)
        num_predicted += 1

    # logs batch_size * seq_len as expected
    logger.info("num predicted=%s", num_predicted)


def _create_labels_for_embeddings_projector(checkpoint_path, hyperparams):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # the metadata file should contain the mapping from data to labels
    # in our case its just the vocabulary
    data_path = os.path.join(os.getcwd(), hyperparams.data.data_path)
    metadata_file = os.path.join(data_path, hyperparams.data.vocab_path_features)
    destination = os.path.join(checkpoint_path, PROJECTOR_METADATA_FILE_NAME)
    copy2(metadata_file, destination)
    # the projector config files states what name is given to the metadata file
    copy2(PROJECTOR_CONFIG_FILE, checkpoint_path)

    # TensorFlow creates a different folder for the eval
    eval_checkpoint = os.path.join(checkpoint_path, "eval")
    if not os.path.exists(eval_checkpoint):
        os.makedirs(eval_checkpoint)
    eval_destination = os.path.join(eval_checkpoint, PROJECTOR_METADATA_FILE_NAME)
    copy2(metadata_file, eval_destination)
    copy2(PROJECTOR_CONFIG_FILE, eval_checkpoint)


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
        create_model (func): creates the model, the function arguments must be (features, mode, params, shared_params)
            where feature is the input_fn (in our case the input pipeline from tf.data) mode is an instance of
            tf.estimator.ModeKeys and params is a dict containing hyperparams used in model.
        create_loss (func): defines the loss, receives as args the model in a dict from create model,
            the labels and hyperparams.
        create_optimizer (func): defines the optimizer, receives as args the loss dict from create loss and hyperparams.
            Returns the train_op and a dictionary that can contain additional info for hooks.
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
    # Create labels for the embeddings projector
    _create_labels_for_embeddings_projector(checkpoint_path=checkpoint_path, hyperparams=hyperparams)

    # Create the datasets
    train_dataset = _create_input_fn(tf_record_path=train_tf_record_path,
                                     hyperparams=hyperparams)
    validation_dataset = _create_input_fn(tf_record_path=valid_tf_record_path,
                                          hyperparams=hyperparams)
    test_dataset = _create_input_fn(tf_record_path=test_tf_record_path,
                                    hyperparams=hyperparams)

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
    start_from_checkpoint = shared_hyperparams.train.get_or_default(key='start_from_experiment', default=None)
    ws = None
    if start_from_checkpoint:
        start_path = os.path.join(shared_hyperparams.data.save_path, start_from_checkpoint)
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=start_path)
    # Create the estimator itself
    estimator = tf.estimator.Estimator(model_fn=estimator_spec,
                                       config=config,
                                       params=hyperparams,
                                       warm_start_from=ws)

    for i in range(num_epochs):
        logger.info("starting training epoch #%s", i + 1)
        # Train and evaluate
        _train_estimator(estimator=estimator,
                         dataset=train_dataset,
                         tf_record_path=train_tf_record_path,
                         steps=epoch_size_train)
        _evaluate_estimator(estimator=estimator,
                            dataset=validation_dataset,
                            tf_record_path=valid_tf_record_path,
                            steps=epoch_size_valid)
        logger.info("finished training epoch #%s", i + 1)
        if EarlyStoppingHook.should_stop:
            logger.info("early stopping detected. stopping training on epoch #%s", i + 1)
            break

    _evaluate_estimator(estimator=estimator,
                        dataset=test_dataset,
                        tf_record_path=test_tf_record_path,
                        steps=epoch_size_test)


def predict_with_model(create_model, shared_hyperparams, hyperparams, checkpoint_path):
    estimator_spec = _create_tf_estimator_spec(create_model=create_model,
                                               create_loss=None,
                                               create_optimizer=None,
                                               shared_hyperparams=shared_hyperparams)

    estimator = tf.estimator.Estimator(model_fn=estimator_spec,
                                       model_dir=checkpoint_path,
                                       params=hyperparams)

    _predict_estimator(estimator=estimator, shared_hyperparams=shared_hyperparams, hyperparams=hyperparams)
