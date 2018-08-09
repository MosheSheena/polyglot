import tensorflow as tf

from time import gmtime, strftime
import time
import numpy as np
import os

from rnnlm.utils.hyperparams import load_params
from rnnlm.models.lstm_fast.model import create_model
from rnnlm.models.lstm_fast.loss import create_loss
from rnnlm.models.lstm_fast.optimizer import create_optimizer
from rnnlm.models.lstm_fast import io_service


def run_epoch(session, model, losses, hyperparams, input_pipeline, eval_op=None, verbose=False):
    """
    Runs the model on the given data
    Args:
        session: (tf.Session)
        model: (dict) name_of_tensor -> tensor
        losses: (dict) name_of_loss -> loss_tensor
        hyperparams: (Dict2Obj)
        input_pipeline: (tf.Iterator) the iterator from the tf.data.Dataset
        eval_op: (tf.Tensor) the tensor operation to execute after building the graph and the loss - optional
        verbose: (bool) print metrics after each batch

    Returns:
        The avg loss (perplexity) of the epoch
    """
    start_time = time.time()
    costs = 0.0
    iters = 0
    step = 0
    state = session.run(model["initial_state"])

    fetches = {
        "cost": losses["cost"],
        "final_state": model["final_state"],
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    while True:
        try:
            session.run(input_pipeline)
            feed_dict = {}
            for i, (c, h) in enumerate(model["initial_state"]):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            state = vals["final_state"]

            costs += cost
            iters += hyperparams.arch.hidden_layer_depth

            # if verbose and step % (epoch_size // 10) == 10:
            if verbose and step % 10 == 0:
                print("perplexity: %.3f speed: %.0f wps" %
                      (np.exp(costs / iters),
                       iters * hyperparams.train.batch_size / (time.time() - start_time)))
            step += 1
        except tf.errors.OutOfRangeError:
            break

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

    if hyperparams.problem.convert_raw_to_tf_records:
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

    """raw_data = reader.rnnlm_raw_data(abs_data_path, abs_vocab_path)
    train_data, valid_data, test_data, _, word_map, _ = raw_data

    size_train = len(train_data)
    size_valid = len(valid_data)
    size_test = len(test_data)

    epoch_size_train = ((size_train // hyperparams.train.batch_size) - 1) // hyperparams.arch.hidden_layer_depth
    epoch_size_valid = ((size_valid // hyperparams.train.batch_size) - 1) // hyperparams.arch.hidden_layer_depth
    epoch_size_test = ((size_test // hyperparams.train.batch_size) - 1) // hyperparams.arch.hidden_layer_depth"""

    with tf.Graph().as_default():

        next_iter_train = io_service.load_tf_records(tf_record_path=train_tf_record_path,
                                                     batch_size=hyperparams.train.batch_size,
                                                     seq_len=hyperparams.arch.hidden_layer_depth)
        next_iter_valid = io_service.load_tf_records(tf_record_path=valid_tf_record_path,
                                                     batch_size=hyperparams.train.batch_size,
                                                     seq_len=hyperparams.arch.hidden_layer_depth)
        next_iter_test = io_service.load_tf_records(tf_record_path=test_tf_record_path,
                                                    batch_size=hyperparams.train.batch_size,
                                                    seq_len=hyperparams.arch.hidden_layer_depth)

        # each call of session.run(next_iter) returns (x, y) where each one is a tensor of shape [batch_size, seq_len]

        initializer = tf.random_uniform_initializer(-hyperparams.train.w_init_scale,
                                                    hyperparams.train.w_init_scale)

        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                training_model = create_model(input_tensor=next_iter_train[0],
                                              mode=None,
                                              hyperparams=hyperparams,
                                              is_training=True)
                training_losses, training_metrics = create_loss(model=training_model,
                                                                labels=next_iter_train[1],
                                                                mode=None,
                                                                hyperparams=hyperparams)
                train_op, lr_update_op, current_lr, new_lr = create_optimizer(model=training_model,
                                                                              losses=training_losses,
                                                                              is_training=True,
                                                                              hyperparams=hyperparams)
            tf.summary.scalar("Training Loss", training_losses["cost"])
            tf.summary.scalar("Learning Rate", current_lr)

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                valid_model = create_model(input_tensor=next_iter_valid[0],
                                           mode=None,
                                           hyperparams=hyperparams,
                                           is_training=False)
                valid_losses, valid_metrics = create_loss(model=valid_model,
                                                          labels=next_iter_valid[1],
                                                          mode=None,
                                                          hyperparams=hyperparams)
                create_optimizer(model=valid_model, losses=valid_losses, is_training=False, hyperparams=hyperparams)
            tf.summary.scalar("Validation Loss", valid_losses["cost"])

        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                test_model = create_model(input_tensor=next_iter_test[0],
                                          mode=None,
                                          hyperparams=hyperparams,
                                          is_training=False)
                test_losses, test_metrics = create_loss(model=test_model,
                                                        labels=next_iter_test[1],
                                                        mode=None,
                                                        hyperparams=hyperparams)
                create_optimizer(test_model, test_losses, False, hyperparams)
        tf.summary.scalar("Test Loss", test_losses["cost"])

        sv = tf.train.Supervisor(logdir=abs_save_path)
        with sv.managed_session() as session:
            for i in range(hyperparams.train.num_epochs):
                lr_decay = hyperparams.train.learning_rate.decay ** max(
                    i + 1 - hyperparams.train.learning_rate.decay_max_factor, 0.0)
                assign_lr(session, lr_update_op, hyperparams.train.learning_rate.start_value * lr_decay, new_lr)
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(current_lr)))
                train_perplexity = run_epoch(session,
                                             training_model,
                                             training_losses,
                                             hyperparams=hyperparams,
                                             input_pipeline=next_iter_train,
                                             eval_op=train_op,
                                             verbose=True)

                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session,
                                             valid_model,
                                             valid_losses,
                                             hyperparams=hyperparams,
                                             input_pipeline=next_iter_valid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session,
                                        test_model,
                                        test_losses,
                                        hyperparams=hyperparams,
                                        input_pipeline=next_iter_test)
            print("Test Perplexity: %.3f" % test_perplexity)
            if hyperparams.train.save_path:
                print("Saving model to %s." % abs_save_path)
                sv.saver.save(session, abs_save_path)
    print(strftime("end time: %Y-%m-%d %H:%M:%S", gmtime()))


if __name__ == "__main__":
    main()
