import logging.config

import yaml

from rnnlm import config as rnnlm_config
from rnnlm.utils.tf_io import extractor
from rnnlm.utils.tf_io.preprocessor.preprocess import preprocess_elements_with_vocab

logging.config.dictConfig(yaml.load(open(rnnlm_config.LOGGING_CONF_PATH, 'r')))
logger = logging.getLogger('rnnlm.models.pos_classifier.pre_training')


def main(raw_files,
         tf_record_outputs,
         features_vocab,
         labels_vocab,
         shared_hyperparams,
         hyperparams):

    logger.info("converting POS TF records")
    seq_len = shared_hyperparams.arch.sequence_length
    extract_fn = extractor.extract_words_and_their_pos_tags

    for raw_path, tf_record_path in zip(raw_files, tf_record_outputs):
        with open(raw_path, 'r') as f:

            data_extractor = extract_fn(opened_file=f, seq_len=seq_len)

            preprocess_elements_with_vocab(extractor=data_extractor,
                                           abs_vocab_path_features=features_vocab,
                                           abs_vocab_path_labels=labels_vocab,
                                           abs_output_tf_record_path=tf_record_path)
