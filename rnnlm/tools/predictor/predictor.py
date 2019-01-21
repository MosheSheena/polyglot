from rnnlm.utils.estimator.estimator import create_prediction_estimator
from rnnlm.utils.tf_io.io_service import create_dataset_from_tensor

import argparse


def predict(create_model, feature_name_to_value_dict, shared_hyperparams, hyperparams):
    predictor = create_prediction_estimator(create_model=create_model,
                                            checkpoint_path=shared_hyperparams.problem.save_path,
                                            shared_hyperparams=shared_hyperparams,
                                            hyperparams=hyperparams)

    predict_input = create_dataset_from_tensor(name_to_tensor_dict=feature_name_to_value_dict,
                                               batch_size=hyperparams.train.batch_size)

    predictions = predictor.predict(input_fn=lambda: predict_input)

    class_id = predictions['class_ids'][0]
    probability = predictions['probabilities'][class_id]


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


if __name__ == '__main__':
    main()
