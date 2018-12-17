class Task:
    """
    Contains all the info needed for training a model.
    """

    def __init__(self,
                 name,
                 create_loss,
                 create_optimizer,
                 train_tf_record_path,
                 valid_tf_record_path,
                 test_tf_record_path,
                 hyperparams,
                 training_hooks=None,
                 evaluation_hooks=None):
        self.name = name
        self.create_loss = create_loss
        self.create_optimizer = create_optimizer
        self.train_tf_record_path = train_tf_record_path
        self.valid_tf_record_path = valid_tf_record_path
        self.test_tf_record_path = test_tf_record_path
        self.hyperparams = hyperparams
        self.training_hooks = training_hooks
        self.evaluation_hooks = evaluation_hooks
