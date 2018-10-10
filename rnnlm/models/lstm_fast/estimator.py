import tensorflow as tf
import os

from rnnlm.utils.estimator_hook import EstimatorHook, LearningRateDecayHook
from rnnlm.models.lstm_fast.io_service import load_dataset


def create_tf_estimator_spec(create_model, create_loss, create_optimizer):
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


def create_input_fn(tf_record_path, hyperparams):

    def input_fn():
        # use global_step to skip records that were already seen by the model
        return load_dataset(tf_record_path=tf_record_path,
                            batch_size=hyperparams.train.batch_size,
                            seq_len=hyperparams.arch.hidden_layer_depth,
                            skip_first_n=tf.train.get_global_step())

    return input_fn


def train_and_evaluate_model(create_model,
                             create_loss,
                             create_optimizer,
                             hyperparams):
    """
    Invoke tf.estimator with the passed args

    Args:
        create_model: a function that creates the model, the function arguments must be (features, mode, params)
            where feature is the input_fn (in our case the input pipeline from tf.data) mode is an instance of
            tf.estimator.ModeKeys and params is a dict containing hyperparams used in model
        create_loss:
        create_optimizer:
        hyperparams:

    Returns:
        None
    """

    abs_data_path = os.path.join(os.getcwd(), hyperparams.problem.data_path)
    abs_vocab_path = os.path.join(os.getcwd(), hyperparams.problem.vocab_path)
    abs_save_path = os.path.join(os.getcwd(), hyperparams.train.save_path)
    abs_tf_record_path = os.path.join(os.getcwd(), hyperparams.problem.tf_records_path)

    train_tf_record_path = os.path.join(abs_tf_record_path, "train.tfrecord")
    valid_tf_record_path = os.path.join(abs_tf_record_path, "valid.tfrecord")
    test_tf_record_path = os.path.join(abs_tf_record_path, "test.tfrecord")

    train_dataset = create_input_fn(tf_record_path=train_tf_record_path, hyperparams=hyperparams)

    validation_dataset = create_input_fn(tf_record_path=valid_tf_record_path, hyperparams=hyperparams)

    # Create estimator spec object
    estimator_spec = create_tf_estimator_spec(create_model, create_loss, create_optimizer)

    # Create the estimator itself
    estimator = tf.estimator.Estimator(model_fn=estimator_spec, params=hyperparams)

    # Create train and eval spec

    # TODO - design the support for multi tasks, make it configurable to receive different input each batch / epoch
    # we may define the training steps here but it then will create the estimator spec which might
    # not be necessary, the alternative is to define the dataset to raise OutOfRange depending on epoch size
    # if we do not define max_steps, the model will train forever on the same dataset
    train_spec = tf.estimator.TrainSpec(input_fn=train_dataset,
                                        max_steps=hyperparams.train.epoch_size_train)
    eval_spec = tf.estimator.EvalSpec(input_fn=validation_dataset)

    # Train and evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
