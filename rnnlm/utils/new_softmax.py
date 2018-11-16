import tensorflow as tf


def new_softmax(labels, logits):
    """
    this new "softmax" function we show can train a "self-normalized" RNNLM
    where the sum of the output is automatically (close to) 1.0
    which saves a lot of computation for lattice-rescoring
    """
    target = tf.reshape(labels, [-1])
    f_logits = tf.exp(logits)

    # this is the negative part of the objf
    row_sums = tf.reduce_sum(f_logits, 1)

    t2 = tf.expand_dims(target, 1)
    range = tf.cast(tf.expand_dims(tf.range(tf.shape(target)[0]), 1), dtype=tf.int64)
    ind = tf.concat([range, t2], 1)
    res = tf.gather_nd(logits, ind)

    return -res + row_sums - 1