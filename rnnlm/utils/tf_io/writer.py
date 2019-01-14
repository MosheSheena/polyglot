import tensorflow as tf


def _int64_feature_wrap(int_values):
    """
    This wraps tf.train.feature.
    This function in used in the process of writing tf records
    Args:
        int_values: (list) a list of integers

    Returns:
        (tf.train.Feature)
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int_values))


def _float64_feature_wrap(float_values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=float_values))


def _bytes_feature_wrap(bytes_values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes_values))


def write_tf_records(gen_words,
                     destination_path,
                     preprocessor_feature_fn=None,
                     preprocessor_feature_params=None,
                     preprocessor_label_fn=None,
                     preprocessor_label_params=None):
    """
    Writes the data in a tf record format
    Args:
        gen_words (generator):  each call to next(gen_word) will yield a tuple of x, y
        destination_path (str): where to write the tf records files
        preprocessor_feature_fn (func):  preprocessor function for feature before writing
        preprocessor_feature_params (args): for preprocessor function, this will be unpacked
        like this fn(x_feature, unpacked_args)
        preprocessor_label_fn (func):  preprocessor function for label before writing
        preprocessor_label_params (args): for preprocessor function, this will be unpacked
        like this fn(y_label, unpacked_args)
    Returns:
        None
    """
    with tf.python_io.TFRecordWriter(destination_path) as writer:

        for x, y in gen_words:

            if preprocessor_feature_fn is not None:
                x = preprocessor_feature_fn(x, *preprocessor_feature_params)
            # TODO support float and byte features
            x_feature = _int64_feature_wrap(int_values=x)

            if preprocessor_label_fn is not None:
                y = preprocessor_label_fn(y, *preprocessor_label_params)
            # TODO support float and byte features
            y_label = _int64_feature_wrap(int_values=y)
            feature_dict = {
                "x": x_feature,
                "y": y_label
            }
            words = tf.train.Features(feature=feature_dict)

            example = tf.train.Example(features=words)

            writer.write(example.SerializeToString())
