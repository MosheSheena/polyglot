import tensorflow as tf
from rnnlm.utils.hyperparams import load_params
from rnnlm.models.lstm_fast.model import create_model
from rnnlm.models.lstm_fast.loss import create_loss
from rnnlm.models.lstm_fast.optimizer import create_optimizer
from time import gmtime, strftime
from rnnlm.models.lstm_fast import reader
import time
import numpy as np
import os


class Config(object):
    """Small config."""

    def __init__(self, hyperparameters):
        self.init_scale = hyperparameters.train.w_init_scale
        self.learning_rate = hyperparameters.train.learning_rate.start_value
        self.max_grad_norm = hyperparameters.train.max_grad_norm
        self.num_layers = hyperparameters.arch.num_hidden_layers
        self.num_steps = hyperparameters.arch.hidden_layer_depth
        self.hidden_size = hyperparameters.arch.hidden_layer_size
        self.max_epoch = hyperparameters.train.learning_rate.decay_max_factor
        self.max_max_epoch = hyperparameters.train.num_epochs
        self.keep_prob = hyperparameters.arch.keep_prob
        self.lr_decay = hyperparameters.train.learning_rate.decay
        self.batch_size = hyperparameters.train.batch_size
        self.vocab_size = hyperparameters.problem.vocab_size


class RnnlmInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.rnnlm_producer(
            data, batch_size, num_steps, name=name)


def get_config(hyperparameters):
    return Config(hyperparameters)


def run_epoch(session, model, losses, rnnlm_input, eval_op=None, verbose=False):
    """Runs the model on the given data."""
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


def assign_lr(session, lr_update, lr_value, _new_lr):
    return session.run(lr_update, feed_dict={_new_lr: lr_value})


def main():
    hyperparams = load_params(os.path.join(os.getcwd(), "rnnlm/models/lstm_fast/hyperparameters.json"))
    print(strftime("start time: %Y-%m-%d %H:%M:%S", gmtime()))

    if not hyperparams.problem.data_path:
        raise ValueError("Must set data_path hyperparameters.json")

    abs_data_path = os.path.join(os.getcwd(), hyperparams.problem.data_path)
    abs_vocab_path = os.path.join(os.getcwd(), hyperparams.problem.vocab_path)
    abs_save_path = os.path.join(os.getcwd(), hyperparams.train.save_path)
    raw_data = reader.rnnlm_raw_data(abs_data_path, abs_vocab_path)
    train_data, valid_data, test_data, _, word_map, _ = raw_data

    config = get_config(hyperparams)
    eval_config = get_config(hyperparams)
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = RnnlmInput(config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = create_model(input_tensor=None,
                                 mode=None,
                                 hyperparams=hyperparams,
                                 is_training=True,
                                 rnnlm_input=train_input)
                m_losses, m_metrics = create_loss(model=m,
                                                  labels=train_input.targets,
                                                  mode=None,
                                                  hyperparams=hyperparams,
                                                  rnnlm_input=train_input)
                m_train_op, m_lr_update_op, current_lr, new_lr = create_optimizer(model=m,
                                                                                  losses=m_losses,
                                                                                  is_training=True,
                                                                                  hyperparams=hyperparams)
            tf.summary.scalar("Training Loss", m_losses["cost"])
            tf.summary.scalar("Learning Rate", current_lr)

        with tf.name_scope("Valid"):
            valid_input = RnnlmInput(config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = create_model(input_tensor=None,
                                      mode=None,
                                      hyperparams=hyperparams,
                                      is_training=False,
                                      rnnlm_input=valid_input)
                mvalid_losses, mvalid_metrics = create_loss(model=mvalid,
                                                            labels=valid_input.targets,
                                                            mode=None,
                                                            hyperparams=hyperparams,
                                                            rnnlm_input=valid_input)
                create_optimizer(model=mvalid, losses=mvalid_losses, is_training=False, hyperparams=hyperparams)
            tf.summary.scalar("Validation Loss", mvalid_losses["cost"])

        # added 29/04/18
        with tf.name_scope("Test"):
            test_input = RnnlmInput(config=config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = create_model(input_tensor=None,
                                     mode=None,
                                     hyperparams=hyperparams,
                                     is_training=False,
                                     rnnlm_input=test_input)
                mtest_losses, mtest_metrics = create_loss(model=mtest,
                                                          labels=test_input.targets,
                                                          mode=None,
                                                          hyperparams=hyperparams,
                                                          rnnlm_input=test_input)
                create_optimizer(mtest, mtest_losses, False, hyperparams)
        tf.summary.scalar("Test Loss", mtest_losses["cost"])
        # end of text edit 29/04/18

        sv = tf.train.Supervisor(logdir=abs_save_path)
        with sv.managed_session() as session:
            for i in range(hyperparams.train.num_epochs):
                lr_decay = hyperparams.train.learning_rate.decay ** max(i + 1 - hyperparams.train.learning_rate.decay_max_factor, 0.0)
                new_lr_op = assign_lr(session, m_lr_update_op, hyperparams.train.learning_rate.start_value * lr_decay, new_lr)
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(current_lr)))
                train_perplexity = run_epoch(session, m, m_losses, train_input, eval_op=m_train_op,
                                             verbose=True)

                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid, mvalid_losses, valid_input)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest, mtest_losses, test_input)
            print("Test Perplexity: %.3f" % test_perplexity)
            if hyperparams.train.save_path:

                print("Saving model to %s." % abs_save_path)
                sv.saver.save(session, abs_save_path)
    print(strftime("end time: %Y-%m-%d %H:%M:%S", gmtime()))


if __name__ == "__main__":
    main()
