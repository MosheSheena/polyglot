from rnnlm.utils.estimator.estimator import create_prediction_estimator
from rnnlm.utils.hyperparams import load_params
from rnnlm.models.lstm_fast.model import create_model

import argparse


def predict(create_model, feature_name_to_value_dict, shared_hyperparams, hyperparams):
    predictor, data = create_prediction_estimator(create_model=create_model,
                                                  prediction_dict=feature_name_to_value_dict,
                                                  checkpoint_path=shared_hyperparams.problem.save_path,
                                                  shared_hyperparams=shared_hyperparams,
                                                  hyperparams=hyperparams)

    predictions = predictor.predict(input_fn=lambda: data)

    class_id = next(predictions)['arg_max']
    probability = predictions['softmax']


def main():
    parser = argparse.ArgumentParser(
        description="\nA tool for predicting words/sentences using a language model\n",
        add_help=True
    )

    # Add arguments
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Name of the model to use to predict, must be the same name from the rnnlm/models directory",
        required=True
    )

    words = {"w": ["HELLO"]}
    config = load_params('experiment_config.json')
    shared_params = config.experiments[0].hyperparameters.shared_layer
    lstm_fast_params = config.experiments[0].hyperparameters.lstm_fast
    predict(create_model, words, shared_hyperparams=shared_params, hyperparams=lstm_fast_params)


if __name__ == '__main__':
    main()
