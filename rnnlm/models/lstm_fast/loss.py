import tensorflow as tf

from rnnlm.utils.new_softmax import new_softmax


def data_type(hyperparams):
    return tf.float16 if hyperparams.train.get_or_default(key="use_fp16", default=False) else tf.float32


def create_loss(model, labels, hyperparams):

    metrics = dict()

    batch_size = hyperparams.train.batch_size
    num_steps = labels.get_shape().as_list()[1]

    flat_labels = tf.reshape(labels, [-1])
    loss_vector = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [model["logits"]],
        [flat_labels],
        [tf.ones([batch_size * num_steps], dtype=data_type(hyperparams))],
        softmax_loss_function=new_softmax)
    loss = tf.reduce_sum(loss_vector) / batch_size
    ppl = tf.math.pow([2.0], loss)
    ppl = tf.reshape(ppl, [])
    tf.summary.scalar("perplexity", ppl)
    return loss, metrics
