from rnnlm.models.lstm_fast import train, pre_training
from rnnlm.utils.hyperparams import load_params
import os
from time import gmtime, strftime
import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main__":
    print(strftime("start time: %Y-%m-%d %H:%M:%S", gmtime()))
    hyperparams = load_params(os.path.join(os.getcwd(), "rnnlm/models/lstm_fast/hyperparameters.json"))

    if not hyperparams.problem.data_path:
        raise ValueError("Must set data_path hyperparameters.json")

    if hyperparams.problem.convert_raw_to_tf_records:
        pre_training.main(hyperparams=hyperparams)

    train.main()

    print(strftime("end time: %Y-%m-%d %H:%M:%S", gmtime()))
