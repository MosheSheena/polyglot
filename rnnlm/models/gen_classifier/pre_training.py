import logging.config

import yaml

from rnnlm import config as rnnlm_config
from rnnlm.utils.tf_io import extractor
from rnnlm.utils.tf_io.preprocessor.preprocess import preprocess_elements_with_vocab

logging.config.dictConfig(yaml.load(open(rnnlm_config.LOGGING_CONF_PATH, 'r')))
logger = logging.getLogger('rnnlm.models.gen_classifier.pre_training')


def main(raw_files,
         tf_record_outputs,
         features_vocab,
         labels_vocab,
         shared_hyperparams,
         hyperparams):
    # pre-process for classic training
    logger.info("converting generated data to tf record")

    seq_len = shared_hyperparams.arch.sequence_length
    x_shifts = hyperparams.data.get_or_default(key="num_shifts_x", default=seq_len)

    train_generated = hyperparams.data.train_generated_data_file
    valid_generated = hyperparams.data.valid_generated_data_file
    test_generated = hyperparams.data.test_generated_data_file

    generated_datasets = [train_generated, valid_generated, test_generated]

    for raw_path, tf_record_path, generated_path in zip(raw_files, tf_record_outputs, generated_datasets):
        with open(raw_path, 'r') as f, open(generated_path) as generated_f:
            data_extractor = extractor.extract_real_with_generated_dataset(real_dataset=f,
                                                                           seq_len=seq_len,
                                                                           num_shifts_real=x_shifts,
                                                                           generated_dataset=generated_f,
                                                                           num_shift_generated=x_shifts)

            preprocess_elements_with_vocab(extractor=data_extractor,
                                           abs_vocab_path_features=features_vocab,
                                           abs_vocab_path_labels=labels_vocab,
                                           abs_output_tf_record_path=tf_record_path)
