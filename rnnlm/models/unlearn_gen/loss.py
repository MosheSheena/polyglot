import tensorflow as tf


def data_type(hyperparams):
    return tf.float16 if hyperparams.train.get_or_default(key="use_fp16", default=False) else tf.float32


def create_loss(model, labels, hyperparams):

    batch_size = hyperparams.train.batch_size

    loss_vector = unlearn1minprob_softmax(
        logits=model["logits_gen"],
        labels=labels
        )
    loss = tf.reduce_sum(loss_vector) / batch_size
    loss_dict = dict()

    return loss, loss_dict


def unlearn1minprob_softmax(labels, logits):  # minimize err prob

    res, row_sums, ind, f_logits = softmax_calc(labels, logits)
    log_prob = res - row_sums  # log probabilities:  log((e^zi) / sum(e^zi))
    prob = tf.exp(log_prob)  # real probabilities
    res_comp = tf.log(1 - prob)  # probabilities of being not err

    return - res_comp  # maximize 1-prob(err) i.e. minimizing err prob


def softmax_calc(labels, logits):
    # calculate res=word_logits (z), row_sums=sum(e^z), ind=Word indexes, f_logits=e^z
    target = tf.reshape(labels, [-1])
    f_logits = tf.exp(logits)
    row_sums = tf.reduce_sum(f_logits, 1)  # this is the negative part of the obj

    t2 = tf.expand_dims(target, 1)
    range = tf.expand_dims(tf.range(tf.shape(target)[0]), 1)
    range = tf.cast(range, dtype=tf.int64)
    ind = tf.concat([range, t2], 1)
    res = tf.gather_nd(logits, ind)

    return res, row_sums, ind, f_logits
