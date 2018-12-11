import tensorflow as tf

from rnnlm.utils.new_softmax import new_softmax


def data_type(hyperparams):
    return tf.float16 if hyperparams.train.use_fp16 else tf.float32


def create_loss(model, labels, hyperparams):

    metrics = dict()

    batch_size = hyperparams.train.batch_size
    num_steps = hyperparams.arch.sequence_length
    flat_labels = tf.reshape(labels, [batch_size * num_steps])
    ones = tf.ones([9999 * 1280], dtype=tf.int64)
    combine_ones = tf.reshape(tf.concat([flat_labels, ones], 0), [1280, 10000])
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels,
            logits=model['logits']
        )
    )
    return loss, metrics
