import tensorflow as tf


def data_type(hyperparams):
    return tf.float16 if hyperparams.train.get_or_default(key="use_fp16", default=False) else tf.float32


def create_loss(model, labels, hyperparams):
    metrics = dict()

    batch_size = hyperparams.train.batch_size
    num_steps = labels.get_shape().as_list()[1]

    loss_vector = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [model["logits_gen"]],
        [tf.reshape(labels, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type(hyperparams))]
        )
    loss = tf.reduce_sum(loss_vector) / batch_size
    return loss, metrics
