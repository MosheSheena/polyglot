import tensorflow as tf


def create_optimizer(losses, hyperparams):

    _lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(losses["cost"], tvars),
                                      hyperparams.train.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(_lr)
    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step()
    )

    _new_lr = tf.placeholder(tf.float32,
                             shape=[],
                             name="new_learning_rate")
    lr_update = tf.assign(_lr, _new_lr)

    # TODO - design a way to pass the args below in order to update learning rate

    losses["lr"] = _lr
    losses["lr_update_op"] = lr_update
    losses["new_lr"] = _new_lr
    losses["train_op"] = train_op

    return train_op, lr_update, _lr, _new_lr
