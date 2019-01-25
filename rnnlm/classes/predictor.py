from rnnlm.utils.estimator.estimator import predict_with_model


class Predictor:

    def __init__(self, create_model, checkpoint_path, shared_hyperparams, create_model_hyperparams):
        self.create_model = create_model
        self.checkpoint_path = checkpoint_path
        self.shared_hyperparams = shared_hyperparams
        self.hyperparams = create_model_hyperparams

    def predict(self):
        predict_with_model(create_model=self.create_model,
                           shared_hyperparams=self.shared_hyperparams,
                           hyperparams=self.hyperparams,
                           checkpoint_path=self.checkpoint_path)

