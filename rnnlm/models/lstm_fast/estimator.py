import tensorflow as tf

from rnnlm.utils.estimator_hook import EstimatorHook


def create_tf_estimator_spec(create_model, create_loss, create_optimizer):

    def my_model_fn(features, labels, mode, params):

        # Create a model
        model = create_model(features, mode, params)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=model)

        # Create a loss
        losses, metrics = create_loss(model, labels, params)
        loss = losses["loss"]

        hooks = [EstimatorHook(model=model, losses=losses, hyperparams=params)]

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=hooks)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # Create an optimizer
            # TODO - support learning rate change like legacy model did
            train_op, lr_update_op, current_lr, new_lr = create_optimizer(losses, params)
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


def train_and_evaluate_model(train_dataset,
                             validation_dataset,
                             create_model,
                             create_loss,
                             create_optimizer,
                             hyperparams):
    """
    Invoke tf.estimator with the passed args

    Args:
        train_dataset :
        validation_dataset :
        create_model: a function that creates the model, the function arguments must be (features, mode, params)
            where feature is the input_fn (in our case the input pipeline from tf.data) mode is an instance of
            tf.estimator.ModeKeys and params is a dict containing hyperparams used in model
        create_loss:
        create_optimizer:
        hyperparams:

    Returns:
        None
    """

    # Create estimator spec object
    estimator_spec = create_tf_estimator_spec(create_model, create_loss, create_optimizer)

    # Create the estimator itself
    estimator = tf.estimator.Estimator(model_fn=estimator_spec, params=hyperparams)

    # Create train and eval spec

    # TODO - design the support for multi tasks, make it configurable to receive different input each batch / epoch
    # we may define the training steps here but it then will create the estimator spec which might
    # not be necessary, the alternative is to define the dataset to raise OutOfRange depending on epoch size
    # if we do not define max_steps, the model will train forever on the same dataset
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_dataset,
                                        max_steps=hyperparams.train.epoch_size_train)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: validation_dataset)

    # Train and evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


