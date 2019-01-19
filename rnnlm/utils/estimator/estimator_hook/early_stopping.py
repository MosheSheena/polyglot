import tensorflow as tf


class EarlyStoppingHook(tf.train.SessionRunHook):
    """
    hold the logic whether the estimator should stop the training earlier.
    if this class decides that it should, than the estimator will finish the training batch
    then he will run the evaluation of the validation and test sets and test will exit

    the global variable here is necessary because the manipulation of the estimator
    for supporting transfer and multitask learning requires the indication of the should
    stop global var in several different places

    """

    # indicates whether the training should stop
    # is global because other classes like the Trainer and ExperimentRunner need to know when
    # to stop the training
    should_stop = False

    # how many batches have passed without improvement on the loss
    # is global since we wish to remember the state between creations of this class
    # since the estimator is recreated every time we invoke a training run
    no_improve_counter = 0

    def __init__(self, loss_tensor, threshold, max_steps_without_improvement):
        self.loss_tensor = loss_tensor
        self.before_loss = None
        self.after_loss = None
        self.max_no_improvement = max_steps_without_improvement
        self.threshold = threshold

    def after_create_session(self, session, coord):
        self.before_loss = session.run(self.loss_tensor)

    def before_run(self, run_context):
        fetches = {"loss": self.loss_tensor}

        run_args = tf.train.SessionRunArgs(fetches=fetches)
        return run_args

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):

        results = run_values.results

        self.after_loss = results["loss"]
        if abs(self.after_loss - self.before_loss) <= self.threshold:
            EarlyStoppingHook.no_improve_counter += 1
        else:
            EarlyStoppingHook.no_improve_counter = 0
        if self.should_stop_fn():
            run_context.request_stop()
        self.before_loss = self.after_loss

    def should_stop_fn(self):
        """
        needed for the early stopping of the estimator, if it returns True,
        then the estimator will stop training, else the training continues.
        For some unknown reason, it must not take any arguments, hopefully tis will change.
        But until then, we use a global variable for saving the previous loss and the no improvement
        counter.
        Returns:
            bool indicating whether to stop training or not
        """

        # TODO - print to debug log
        # print("Current loss: {} previous loss: {}".format(self.after_loss, self.before_loss))
        EarlyStoppingHook.should_stop = EarlyStoppingHook.no_improve_counter >= self.max_no_improvement
        # print("should_stop = {}".format(EarlyStoppingHook.should_stop))
        return EarlyStoppingHook.should_stop
