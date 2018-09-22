import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

from time import gmtime, strftime
import time
import numpy as np
import os

from rnnlm.utils.hyperparams import load_params
from rnnlm.models.lstm_fast.model import create_model
from rnnlm.models.lstm_fast.loss import create_loss
from rnnlm.models.lstm_fast.optimizer import create_optimizer
from rnnlm.models.lstm_fast import io_service
from rnnlm.models.lstm_fast.estimator import train_and_evaluate_model


def run_epoch(session, model, losses, hyperparams, epoch_size, eval_op=None, verbose=False):
    """
    Runs the model on the given data
    Args:

        session: (tf.Session)
        model: (dict) name_of_tensor -> tensor
        losses: (dict) name_of_loss -> loss_tensor
        hyperparams: (Dict2Obj)
        epoch_size: (int)
        eval_op: (tf.Tensor) the tensor operation to execute after building the graph and the loss - optional
        verbose: (bool) print metrics after each batch

    Returns:
        The avg loss (perplexity) of the epoch
    """
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model["initial_state"])

    fetches = {
        "cost": losses["cost"],
        "final_state": model["final_state"],
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model["initial_state"]):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += hyperparams.arch.hidden_layer_depth

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * hyperparams.train.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def assign_lr(session, lr_update, lr_value, new_lr):
    """
    Assigns a new learning rate
    Args:
        session: (tf.Session)
        lr_update: (Tensor) tf.assign op tensor
        lr_value: (int) the new value for the learning rate
        new_lr: (Placeholder) a placeholder for the learning rate

    Returns:
        None
    """
    session.run(lr_update, feed_dict={new_lr: lr_value})


def main():
    hyperparams = load_params(os.path.join(os.getcwd(), "rnnlm/models/lstm_fast/hyperparameters.json"))
    print(strftime("start time: %Y-%m-%d %H:%M:%S", gmtime()))

    if not hyperparams.problem.data_path:
        raise ValueError("Must set data_path hyperparameters.json")

    abs_data_path = os.path.join(os.getcwd(), hyperparams.problem.data_path)
    abs_vocab_path = os.path.join(os.getcwd(), hyperparams.problem.vocab_path)
    abs_save_path = os.path.join(os.getcwd(), hyperparams.train.save_path)
    abs_tf_record_path = os.path.join(os.getcwd(), hyperparams.problem.tf_records_path)

    train_tf_record_path = os.path.join(abs_tf_record_path, "train.tfrecord")
    valid_tf_record_path = os.path.join(abs_tf_record_path, "valid.tfrecord")
    test_tf_record_path = os.path.join(abs_tf_record_path, "test.tfrecord")

    if not os.path.exists(abs_tf_record_path):
        os.makedirs(abs_tf_record_path)

    if hyperparams.problem.convert_raw_to_tf_records:
        print("Converting raw data to tfrecord format")
        io_service.raw_to_tf_records(raw_path=os.path.join(abs_data_path, "train"),
                                     tf_record_path=train_tf_record_path,
                                     vocab_path=abs_vocab_path,
                                     seq_len=hyperparams.arch.hidden_layer_depth)
        io_service.raw_to_tf_records(raw_path=os.path.join(abs_data_path, "valid"),
                                     tf_record_path=valid_tf_record_path,
                                     vocab_path=abs_vocab_path,
                                     seq_len=hyperparams.arch.hidden_layer_depth)
        io_service.raw_to_tf_records(raw_path=os.path.join(abs_data_path, "test"),
                                     tf_record_path=test_tf_record_path,
                                     vocab_path=abs_vocab_path,
                                     seq_len=hyperparams.arch.hidden_layer_depth)
        print("Conversion done.")
    with tf.Graph().as_default():
        print("Start training")

        train_and_evaluate_model(create_model=create_model,
                                 create_loss=create_loss,
                                 create_optimizer=create_optimizer,
                                 hyperparams=hyperparams)

        print("End training")
    print(strftime("end time: %Y-%m-%d %H:%M:%S", gmtime()))


if __name__ == "__main__":

    main()
