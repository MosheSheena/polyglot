import tensorflow as tf


def create_optimizer(loss, loss_dict, hyperparams):
    lr = tf.Variable(hyperparams.train.learning_rate.start_value, trainable=False)
    tf.summary.scalar("learning_rate", lr)
    t_vars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, t_vars),
                                      hyperparams.train.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(
        zip(grads, t_vars),
        global_step=tf.train.get_or_create_global_step()
    )

    new_lr = tf.placeholder(tf.float32,
                            shape=[],
                            name="new_learning_rate")
    lr_update_op = tf.assign(lr, new_lr)

    optimizer_params = dict()
    optimizer_params["lr_update_op"] = lr_update_op
    optimizer_params["learning_rate"] = lr
    optimizer_params["new_lr_placeholder"] = new_lr

    return train_op, optimizer_params
