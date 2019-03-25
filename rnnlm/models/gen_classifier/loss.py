import tensorflow as tf


def data_type(hyperparams):
    return tf.float16 if hyperparams.train.get_or_default(key="use_fp16", default=False) else tf.float32


def create_loss(model, labels, hyperparams):

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels,
        logits=model['logits_gen'],
    )

    loss = tf.reduce_mean(loss)
    loss_dict = dict()

    return loss, loss_dict
