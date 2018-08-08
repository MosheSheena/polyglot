import tensorflow as tf
from rnnlm.utils.hyperparams import load_params
from rnnlm.models.lstm_fast.model import create_model
from rnnlm.models.lstm_fast.loss import create_loss
from rnnlm.models.lstm_fast.optimizer import create_optimizer
from time import gmtime, strftime
from rnnlm.models.lstm_fast import reader
from rnnlm.models.lstm_fast import io_service
import time
import numpy as np
import os


class RnnlmInput(object):
    """The input data."""

    def __init__(self, hyperparams, data, name=None):
        self.batch_size = batch_size = hyperparams.train.batch_size
        self.num_steps = num_steps = hyperparams.arch.hidden_layer_depth
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.rnnlm_producer(
            data, batch_size, num_steps, name=name)


def run_epoch(session: tf.Session, model, losses, rnnlm_input, eval_op=None, verbose=False):
    """
    Runs the model on the given data
    Args:
        session: (tf.Session)
        model: (dict) name_of_tensor -> tensor
        losses: (dict) name_of_loss -> loss_tensor
        rnnlm_input: (RnnLMInput) object with the configurations TODO - remove dependency
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

    for step in range(rnnlm_input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model["initial_state"]):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += rnnlm_input.num_steps

        if verbose and step % (rnnlm_input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / rnnlm_input.epoch_size, np.exp(costs / iters),
                   iters * rnnlm_input.batch_size / (time.time() - start_time)))

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

    if hyperparams.problem.convert_raw_to_tf_records:
        io_service.raw_to_tf_records(raw_path=os.path.join(abs_data_path, "train"),
                                     tf_record_path=abs_tf_record_path,
                                     vocab_path=abs_vocab_path,
                                     seq_len=hyperparams.train.num_steps)
        io_service.raw_to_tf_records(raw_path=os.path.join(abs_data_path, "valid"),
                                     tf_record_path=abs_tf_record_path,
                                     vocab_path=abs_vocab_path,
                                     seq_len=hyperparams.train.num_steps)
        io_service.raw_to_tf_records(raw_path=os.path.join(abs_data_path, "test"),
                                     tf_record_path=abs_tf_record_path,
                                     vocab_path=abs_vocab_path,
                                     seq_len=hyperparams.train.num_steps)

    io_service.load_tf_records()
    raw_data = reader.rnnlm_raw_data(abs_data_path, abs_vocab_path)
    train_data, valid_data, test_data, _, word_map, _ = raw_data

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-hyperparams.train.w_init_scale,
                                                    hyperparams.train.w_init_scale)

        with tf.name_scope("Train"):
            train_input = RnnlmInput(hyperparams=hyperparams, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                training_model = create_model(input_tensor=None,
                                              mode=None,
                                              hyperparams=hyperparams,
                                              is_training=True,
                                              rnnlm_input=train_input)
                training_losses, training_metrics = create_loss(model=training_model,
                                                                labels=train_input.targets,
                                                                mode=None,
                                                                hyperparams=hyperparams,
                                                                rnnlm_input=train_input)
                train_op, lr_update_op, current_lr, new_lr = create_optimizer(model=training_model,
                                                                              losses=training_losses,
                                                                              is_training=True,
                                                                              hyperparams=hyperparams)
            tf.summary.scalar("Training Loss", training_losses["cost"])
            tf.summary.scalar("Learning Rate", current_lr)

        with tf.name_scope("Valid"):
            valid_input = RnnlmInput(hyperparams=hyperparams, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                valid_model = create_model(input_tensor=None,
                                           mode=None,
                                           hyperparams=hyperparams,
                                           is_training=False,
                                           rnnlm_input=valid_input)
                valid_losses, valid_metrics = create_loss(model=valid_model,
                                                          labels=valid_input.targets,
                                                          mode=None,
                                                          hyperparams=hyperparams,
                                                          rnnlm_input=valid_input)
                create_optimizer(model=valid_model, losses=valid_losses, is_training=False, hyperparams=hyperparams)
            tf.summary.scalar("Validation Loss", valid_losses["cost"])

        # added 29/04/18
        with tf.name_scope("Test"):
            test_input = RnnlmInput(hyperparams=hyperparams, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                test_model = create_model(input_tensor=None,
                                          mode=None,
                                          hyperparams=hyperparams,
                                          is_training=False,
                                          rnnlm_input=test_input)
                test_losses, test_metrics = create_loss(model=test_model,
                                                        labels=test_input.targets,
                                                        mode=None,
                                                        hyperparams=hyperparams,
                                                        rnnlm_input=test_input)
                create_optimizer(test_model, test_losses, False, hyperparams)
        tf.summary.scalar("Test Loss", test_losses["cost"])
        # end of text edit 29/04/18

        sv = tf.train.Supervisor(logdir=abs_save_path)
        with sv.managed_session() as session:
            for i in range(hyperparams.train.num_epochs):
                lr_decay = hyperparams.train.learning_rate.decay ** max(
                    i + 1 - hyperparams.train.learning_rate.decay_max_factor, 0.0)
                assign_lr(session, lr_update_op, hyperparams.train.learning_rate.start_value * lr_decay, new_lr)
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(current_lr)))
                train_perplexity = run_epoch(session, training_model, training_losses, train_input, eval_op=train_op,
                                             verbose=True)

                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, valid_model, valid_losses, valid_input)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, test_model, test_losses, test_input)
            print("Test Perplexity: %.3f" % test_perplexity)
            if hyperparams.train.save_path:

                print("Saving model to %s." % abs_save_path)
                sv.saver.save(session, abs_save_path)
    print(strftime("end time: %Y-%m-%d %H:%M:%S", gmtime()))


if __name__ == "__main__":
    main()
