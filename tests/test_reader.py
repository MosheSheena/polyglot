import unittest
from rnnlm.models.lstm_fast import reader


class TestReader(unittest.TestCase):

    def test_read_n_shifted_words(self):
        read_n = reader._read_n_shifted_words_gen
        self.assertEqual(list(read_n([], 2, overlap=True)), [])
        self.assertEqual(list(read_n(['1', '2', '3', '4'], 2, overlap=True)),
                         [['1', '2'], ['2', '3'], ['3', '4'], ['4']])
        self.assertEqual(list(read_n(['2'], 1, overlap=True)), [['2']])
        self.assertEqual(list(read_n(['1'], 2, overlap=True)), [['1']])
        self.assertEqual(list(read_n(['1', '2'], 0, overlap=True)), [[]])
        self.assertEqual(list(read_n(['1', '2'], 2, overlap=True)), [['1', '2'], ['2']])

    def test_gen_shifted_word(self):
        shifted_w = reader.gen_shifted_words_with_overlap
        self.assertEqual(list(shifted_w([], 2)), [])
        self.assertEqual(list(shifted_w(['1', '2', '3', '4'], 2)), [(['1', '2'], ['2', '3']), (['2', '3'], ['3', '4'])])
        self.assertEqual(list(shifted_w(['1', '2', '3', '4'], 3)), [(['1', '2', '3'], ['2', '3', '4'])])
        self.assertEqual(list(shifted_w(['1', '2', '3'], 2)), [(['1', '2'], ['2', '3'])])
        self.assertEqual(list(shifted_w(['1', '2', '3', '4', '5'], 3)),
                         [(['1', '2', '3'], ['2', '3', '4']), (['2', '3', '4'], ['3', '4', '5'])])
        self.assertEqual(list(shifted_w(['2'], 1)), [])
        self.assertEqual(list(shifted_w(['1'], 2)), [])
        self.assertEqual(list(shifted_w(['1', '2'], 0)), [])

    def test_read_n_no_overlap(self):
        read_n = reader._read_n_shifted_words_gen
        self.assertEqual(list(read_n([], 2, overlap=False)), [])
        self.assertEqual(list(read_n(['1', '2', '3', '4'], 2, overlap=False)),
                         [['1', '2'], ['3', '4']])
        self.assertEqual(list(read_n(['1', '2', '3'], 2, overlap=False)),
                         [['1', '2'], ['3']])
        self.assertEqual(list(read_n(['2'], 1, overlap=False)), [['2']])
        self.assertEqual(list(read_n(['1'], 2, overlap=False)), [['1']])
        self.assertEqual(list(read_n(['1', '2'], 0, overlap=False)), [[]])
        self.assertEqual(list(read_n(['1', '2'], 2, overlap=False)), [['1', '2']])

    def test_gen_no_overlap_words(self):
        no_overlap_gen = reader.gen_no_overlap_words
        self.assertEqual(list(no_overlap_gen([], 2)), [])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3', '4'], 2)), [(['1', '2'], ['2', '3'])])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3', '4'], 3)), [(['1', '2', '3'], ['2', '3', '4'])])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3'], 2)), [(['1', '2'], ['2', '3'])])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3', '4', '5'], 3)),
                         [(['1', '2', '3'], ['2', '3', '4'])])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3', '4', '5'], 2)),
                         [(['1', '2'], ['2', '3']), (['3', '4'], ['4', '5'])])
        self.assertEqual(list(no_overlap_gen(['2'], 1)), [])
        self.assertEqual(list(no_overlap_gen(['1'], 2)), [])
        self.assertEqual(list(no_overlap_gen(['1', '2'], 0)), [])
