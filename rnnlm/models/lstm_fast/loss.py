import tensorflow as tf

from rnnlm.utils.new_softmax import new_softmax


def data_type(hyperparams):
    return tf.float16 if hyperparams.train.use_fp16 else tf.float32


def create_loss(model, labels, hyperparams):

    metrics = dict()

    batch_size = hyperparams.train.batch_size
    num_steps = hyperparams.arch.sequence_length

    loss_vector = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [model["logits"]],
        [tf.reshape(labels, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type(hyperparams))],
        softmax_loss_function=new_softmax)
    loss = tf.reduce_sum(loss_vector) / batch_size
    return loss, metrics
