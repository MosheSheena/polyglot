import logging

from rnnlm.utils.tf_io import extractor
from rnnlm.utils.tf_io.preprocessor.preprocess import preprocess_elements_with_vocab

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_formatter = formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

fh = logging.FileHandler('pos-classifier_pre_training.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(file_formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
ch.setFormatter(console_formatter)

logger.addHandler(fh)
logger.addHandler(ch)


def main(raw_files,
         tf_record_outputs,
         features_vocab,
         labels_vocab,
         shared_hyperparams,
         hyperparams):

    logger.debug("converting POS TF records")
    seq_len = shared_hyperparams.arch.sequence_length
    extract_fn = extractor.extract_words_and_their_pos_tags

    for raw_path, tf_record_path in zip(raw_files, tf_record_outputs):
        with open(raw_path, 'r') as f:

            data_extractor = extract_fn(opened_file=f, seq_len=seq_len)

            preprocess_elements_with_vocab(extractor=data_extractor,
                                           abs_vocab_path_features=features_vocab,
                                           abs_vocab_path_labels=labels_vocab,
                                           abs_output_tf_record_path=tf_record_path)
