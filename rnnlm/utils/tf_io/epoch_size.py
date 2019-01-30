import tensorflow as tf


def get_epoch_size_from_tf_dataset(tf_dataset):
    tf_it = tf_dataset.make_one_shot_iterator().get_next()
    sess = tf.Session()
    epoch_counter = 0
    while True:
        try:
            sess.run(tf_it)
            epoch_counter += 1
        except tf.errors.OutOfRangeError:
            break

    return epoch_counter


def get_epoch_size_from_generator(gen):
    epoch_counter = 0

    for x, *y in gen:
        epoch_counter += 1

    return epoch_counter
