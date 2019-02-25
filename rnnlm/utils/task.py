class Task:
    """
    Contains all the info needed for training a model.
    """

    def __init__(self,
                 name,
                 create_loss,
                 create_optimizer,
                 hyperparams,
                 training_hooks=None,
                 evaluation_hooks=None):
        self.name = name
        self.create_loss = create_loss
        self.create_optimizer = create_optimizer
        self.hyperparams = hyperparams
        self.training_hooks = training_hooks
        self.evaluation_hooks = evaluation_hooks
