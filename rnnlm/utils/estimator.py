import tensorflow as tf

from rnnlm.utils.estimator_hook import EstimatorHook, LearningRateDecayHook
from rnnlm.utils.tf_io.io_service import load_dataset
from collections import defaultdict

# like tf.train.global_step, only per dataset
dataset_step_counter = defaultdict(int)


def _create_tf_estimator_spec(create_model, create_loss, create_optimizer):
    def my_model_fn(features, labels, mode, params):

        # Create a model
        model = create_model(features, mode, params)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=model)

        # Create a loss
        losses, metrics = create_loss(model, labels, params)
        loss = losses["cost"]

        hooks = [EstimatorHook(model=model, losses=losses, hyperparams=params, mode=mode)]

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=hooks)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Create an optimizer
            train_op, lr_update_op, current_lr, new_lr = create_optimizer(losses, params)

            # Add a hook that decays the learning rate like the legacy model did
            hooks.append(LearningRateDecayHook(lr_update_op=lr_update_op,
                                               current_lr=current_lr,
                                               new_lr=new_lr,
                                               mode=mode,
                                               hyperparams=params))
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op, training_hooks=hooks)

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
        print("path = {}".format(tf_record_path))
        print("dataset_steps = {}".format(dataset_step_counter[tf_record_path]))
        return load_dataset(abs_tf_record_path=tf_record_path,
                            batch_size=hyperparams.train.batch_size,
                            seq_len=hyperparams.arch.hidden_layer_depth,
                            skip_first_n=dataset_step_counter[tf_record_path])

    return input_fn


def train_estimator(estimator, dataset, tf_record_path, steps):
    estimator.train(input_fn=dataset, steps=steps)
    dataset_step_counter[tf_record_path] += steps


def evaluate_estimator(estimator, dataset, tf_record_path, steps):
    estimator.evaluate(input_fn=dataset, steps=steps)
    dataset_step_counter[tf_record_path] += steps


def train_and_evaluate_model(create_model,
                             create_loss,
                             create_optimizer,
                             hyperparams,
                             train_tf_record_path,
                             valid_tf_record_path,
                             test_tf_record_path):
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

    Returns:
        None
    """

    train_dataset = _create_input_fn(tf_record_path=train_tf_record_path, hyperparams=hyperparams)
    validation_dataset = _create_input_fn(tf_record_path=valid_tf_record_path, hyperparams=hyperparams)
    test_dataset = _create_input_fn(tf_record_path=test_tf_record_path, hyperparams=hyperparams)

    # Create estimator spec object
    estimator_spec = _create_tf_estimator_spec(create_model, create_loss, create_optimizer)

    # Create the estimator itself
    estimator = tf.estimator.Estimator(model_fn=estimator_spec,
                                       params=hyperparams)

    for i in range(hyperparams.train.num_epochs):
        # Train and evaluate
        train_estimator(estimator=estimator,
                        dataset=train_dataset,
                        tf_record_path=train_tf_record_path,
                        steps=hyperparams.train.epoch_size_train)
        evaluate_estimator(estimator=estimator,
                           dataset=validation_dataset,
                           tf_record_path=valid_tf_record_path,
                           steps=hyperparams.train.epoch_size_valid)

    evaluate_estimator(estimator=estimator,
                       dataset=test_dataset,
                       tf_record_path=test_tf_record_path,
                       steps=hyperparams.train.epoch_size_test)
