import tensorflow as tf


def data_type(hyperparams):
    return tf.float16 if hyperparams.train.get_or_default(key="use_fp16", default=False) else tf.float32


def create_loss(model, labels, hyperparams):
    metrics = dict()

    batch_size = hyperparams.train.batch_size
    num_steps = labels.get_shape().as_list()[1]

    flat_labels = tf.reshape(labels, [-1])

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels,
        logits=model['logits_gen'],
    )

    loss = tf.reduce_mean(loss)

    return loss, metrics
