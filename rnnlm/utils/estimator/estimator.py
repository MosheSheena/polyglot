from rnnlm.utils.estimator.estimator_hook.perplexity import MeasurePerplexityHook
from rnnlm.utils.tf_io.io_service import load_dataset
from collections import defaultdict

from rnnlm.utils.epoch_size import *


# like tf.train.global_step, only per dataset
dataset_step_counter = defaultdict(int)


def _create_tf_estimator_spec(create_model,
                              create_loss,
                              create_optimizer,
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
        # TODO - maybe convert with Dict2Obj
        estimator_params = dict()
        estimator_params["hyperparameters"] = params
        estimator_params["mode"] = mode

        # Create a model
        model = create_model(features, mode, params)
        estimator_params["model"] = model

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=model)

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
        # TODO replace with logging
        print("path = {}".format(tf_record_path))
        print("dataset_steps = {}".format(dataset_step_counter[tf_record_path]))
        dataset = load_dataset(abs_tf_record_path=tf_record_path,
                               batch_size=hyperparams.train.batch_size,
                               seq_len=hyperparams.arch.sequence_length,
                               skip_first_n=dataset_step_counter[tf_record_path])
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

    train_dataset = _create_input_fn(tf_record_path=train_tf_record_path, hyperparams=hyperparams)
    validation_dataset = _create_input_fn(tf_record_path=valid_tf_record_path, hyperparams=hyperparams)
    test_dataset = _create_input_fn(tf_record_path=test_tf_record_path, hyperparams=hyperparams)

    # Create estimator spec object
    estimator_spec = _create_tf_estimator_spec(create_model=create_model,
                                               create_loss=create_loss,
                                               create_optimizer=create_optimizer,
                                               training_hooks=training_hooks,
                                               evaluation_hooks=evaluation_hooks)

    # Create estimator run config
    summary_steps = hyperparams.train.get_or_default(key="summary_steps", default=100)
    save_checkpoint_steps = hyperparams.train.get_or_default(key="save_checkpoint_steps", default=200)
    keep_checkpoints_max = hyperparams.train.get_or_default(key="keep_checkpoint_max", default=5)
    config = tf.estimator.RunConfig(model_dir=checkpoint_path,
                                    save_summary_steps=summary_steps,
                                    save_checkpoints_steps=save_checkpoint_steps,
                                    keep_checkpoint_max=keep_checkpoints_max)
    # Create the estimator itself
    estimator = tf.estimator.Estimator(model_fn=estimator_spec,
                                       config=config,
                                       params=hyperparams)

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

    _evaluate_estimator(estimator=estimator,
                        dataset=test_dataset,
                        tf_record_path=test_tf_record_path,
                        steps=epoch_size_test)
