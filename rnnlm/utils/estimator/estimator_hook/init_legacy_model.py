import tensorflow as tf


class InitLegacyModelHook(tf.train.SessionRunHook):
    """
    Support initialization made in run epoch of legacy
    TODO - maybe the legacy model does not need this initializations, test it.
    """

    def __init__(self, model, **kwargs):
        self.model = model
        self.state = None

    def after_create_session(self, session, coord):

        # set the state to the initial state
        self.state = session.run(self.model["initial_state"])

    def before_run(self, run_context):
        fetches = {
            "final_state": self.model["final_state"]
        }

        feed_dict = {}
        # update the feed_dict for the current state
        for i, (c, h) in enumerate(self.model["initial_state"]):
            feed_dict[c] = self.state[i].c
            feed_dict[h] = self.state[i].h

        run_args = tf.train.SessionRunArgs(fetches=fetches, feed_dict=feed_dict)
        return run_args

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        results = run_values.results

        # update the state for the next run
        self.state = results["final_state"]
