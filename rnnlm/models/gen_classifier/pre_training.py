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
    x_shifts = hyperparams.problem.get_or_default(key="num_shifts_x", default=seq_len)
    extract_fn = extractor.extract_x_y_words_with_x_shifting_by_n_each_yield

    for raw_path, tf_record_path in zip(raw_files, tf_record_outputs):
        with open(raw_path, 'r') as f:

            data_extractor = extract_fn(opened_file=f, seq_len=seq_len, n=x_shifts)

            preprocess_elements_with_vocab(extractor=data_extractor,
                                           abs_vocab_path_features=features_vocab,
                                           abs_vocab_path_labels=labels_vocab,
                                           abs_output_tf_record_path=tf_record_path)

