import unittest

import rnnlm.utils.tf_io.generator


class TestReader(unittest.TestCase):

    def test_gen_n_shifted_words(self):
        read_n = rnnlm.utils.tf_io.generator._gen_n_words
        self.assertEqual(list(read_n([], 2, overlap=True)), [])
        self.assertEqual(list(read_n(['1', '2', '3', '4'], 2, overlap=True)), [['1', '2'], ['2', '3'], ['3', '4']])
        self.assertEqual(list(read_n(['1', '2', '3', '4'], 3, overlap=True)), [['1', '2', '3'], ['2', '3', '4']])
        self.assertEqual(list(read_n(['2'], 1, overlap=True)), [['2']])
        self.assertEqual(list(read_n(['1'], 2, overlap=True)), [])
        self.assertEqual(list(read_n(['1', '2'], 0, overlap=True)), [])
        self.assertEqual(list(read_n(['1', '2'], 2, overlap=True)), [['1', '2']])
        self.assertEqual(list(read_n(['1', '2', '3'], 2, overlap=True)), [['1', '2'], ['2', '3']])
        self.assertEqual(list(read_n(['1', '2', '3'], 3, overlap=True)), [['1', '2', '3']])
        self.assertEqual(
            list(read_n(['1', '2', '3', '4', '5', '6', '7', '8'], 3, overlap=True)),
            [['1', '2', '3'], ['2', '3', '4'], ['3', '4', '5'], ['4', '5', '6'], ['5', '6', '7'], ['6', '7', '8']])
        self.assertEqual(
            list(read_n(['1', '2', '3', '4', '5', '6', '7', '8'], 4, overlap=True)),
            [['1', '2', '3', '4'], ['2', '3', '4', '5'], ['3', '4', '5', '6'], ['4', '5', '6', '7'], ['5', '6', '7', '8']])
        self.assertEqual(list(read_n(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'],
                                     5, overlap=True)),
                         [['1', '2', '3', '4', '5'], ['2', '3', '4', '5', '6'], ['3', '4', '5', '6', '7'],
                          ['4', '5', '6', '7', '8'], ['5', '6', '7', '8', '9'], ['6', '7', '8', '9', '10'],
                          ['7', '8', '9', '10', '11'], ['8', '9', '10', '11', '12'], ['9', '10', '11', '12', '13']])

    def test_gen_n_no_overlap(self):
        read_n = rnnlm.utils.tf_io.generator._gen_n_words
        self.assertEqual(list(read_n([], 2, overlap=False)), [])
        self.assertEqual(list(read_n(['1', '2', '3', '4'], 2, overlap=False)), [['1', '2'], ['3', '4']])
        self.assertEqual(list(read_n(['1', '2', '3', '4'], 3, overlap=False)), [['1', '2', '3']])
        self.assertEqual(list(read_n(['2'], 1, overlap=False)), [['2']])
        self.assertEqual(list(read_n(['1'], 2, overlap=False)), [])
        self.assertEqual(list(read_n(['1', '2'], 0, overlap=False)), [])
        self.assertEqual(list(read_n(['1', '2'], 2, overlap=False)), [['1', '2']])
        self.assertEqual(list(read_n(['1', '2', '3'], 2, overlap=False)), [['1', '2']])
        self.assertEqual(list(read_n(['1', '2', '3'], 3, overlap=False)), [['1', '2', '3']])
        self.assertEqual(
            list(read_n(['1', '2', '3', '4', '5', '6', '7', '8'], 3, overlap=False)),
            [['1', '2', '3'], ['4', '5', '6']])
        self.assertEqual(
            list(read_n(['1', '2', '3', '4', '5', '6', '7', '8'], 4, overlap=False)),
            [['1', '2', '3', '4'], ['5', '6', '7', '8']])
        self.assertEqual(list(read_n(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'],
                                     5, overlap=False)),
                         [['1', '2', '3', '4', '5'], ['6', '7', '8', '9', '10']])
        self.assertEqual(list(read_n(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
                                     6, overlap=False)),
                         [['1', '2', '3', '4', '5', '6'], ['7', '8', '9', '10', '11', '12']])

    def test_gen_n_with_num_overlap(self):
        read_n = rnnlm.utils.tf_io.generator._gen_n_words
        self.assertEqual(list(read_n([], 2, overlap=True, num_elements_to_overlap=1)), [])
        self.assertEqual(list(read_n(['1', '2', '3', '4'], 2, overlap=True, num_elements_to_overlap=1)),
                         [['1', '2'], ['2', '3'], ['3', '4']])
        self.assertEqual(list(read_n(['1', '2', '3', '4'], 3, overlap=True, num_elements_to_overlap=1)),
                         [['1', '2', '3']])
        self.assertEqual(list(read_n(['2'], 1, overlap=True, num_elements_to_overlap=1)), [['2']])
        self.assertEqual(list(read_n(['1'], 2, overlap=True, num_elements_to_overlap=1)), [])
        self.assertEqual(list(read_n(['1', '2'], 0, overlap=True, num_elements_to_overlap=1)), [])
        self.assertEqual(list(read_n(['1', '2'], 2, overlap=True, num_elements_to_overlap=2)), [['1', '2']])
        self.assertEqual(list(read_n(['1', '2', '3'], 2, overlap=True, num_elements_to_overlap=1)),
                         [['1', '2'], ['2', '3']])
        self.assertEqual(list(read_n(['1', '2', '3'], 3, overlap=True, num_elements_to_overlap=1)), [['1', '2', '3']])
        self.assertEqual(
            list(read_n(['1', '2', '3', '4', '5', '6', '7', '8'], 3, overlap=True, num_elements_to_overlap=2)),
            [['1', '2', '3'], ['2', '3', '4'], ['3', '4', '5'], ['4', '5', '6'], ['5', '6', '7'], ['6', '7', '8']])
        self.assertEqual(
            list(read_n(['1', '2', '3', '4', '5', '6', '7', '8'], 3, overlap=True, num_elements_to_overlap=3)),
            [['1', '2', '3']])
        self.assertEqual(
            list(read_n(['1', '2', '3', '4', '5', '6', '7', '8'], 4, overlap=True, num_elements_to_overlap=3)),
            [['1', '2', '3', '4'], ['2', '3', '4', '5'], ['3', '4', '5', '6'], ['4', '5', '6', '7'], ['5', '6', '7', '8']])
        self.assertEqual(list(read_n(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'],
                                     5, overlap=True, num_elements_to_overlap=3)),
                         [['1', '2', '3', '4', '5'], ['3', '4', '5', '6', '7'], ['5', '6', '7', '8', '9'],
                          ['7', '8', '9', '10', '11'], ['9', '10', '11', '12', '13']])
        self.assertEqual(list(read_n(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
                                     6, overlap=True, num_elements_to_overlap=2)),
                         [['1', '2', '3', '4', '5', '6'], ['5', '6', '7', '8', '9', '10'], ['9', '10', '11', '12', '13', '14']])

    def test_gen_shifted_word(self):
        shifted_w = rnnlm.utils.tf_io.generator.gen_shifted_words_with_overlap
        self.assertEqual(list(shifted_w([], 2)), [])
        self.assertEqual(list(shifted_w(['1', '2', '3', '4'], 2)), [(['1', '2'], ['2', '3']), (['2', '3'], ['3', '4'])])
        self.assertEqual(list(shifted_w(['1', '2', '3', '4'], 3)), [(['1', '2', '3'], ['2', '3', '4'])])
        self.assertEqual(list(shifted_w(['2'], 1)), [])
        self.assertEqual(list(shifted_w(['1'], 2)), [])
        self.assertEqual(list(shifted_w(['1', '2'], 0)), [])
        self.assertEqual(list(shifted_w(['1', '2'], 1)), [(['1'], ['2'])])
        self.assertEqual(list(shifted_w(['1', '2'], 2)), [])
        self.assertEqual(list(shifted_w(['1', '2', '3'], 1)), [(['1'], ['2']), (['2'], ['3'])])
        self.assertEqual(list(shifted_w(['1', '2', '3'], 2)), [(['1', '2'], ['2', '3'])])
        self.assertEqual(list(shifted_w(['1', '2', '3'], 3)), [])
        self.assertEqual(list(shifted_w(['1', '2', '3', '4', '5'], 3)),
                         [(['1', '2', '3'], ['2', '3', '4']), (['2', '3', '4'], ['3', '4', '5'])])
        self.assertEqual(
            list(shifted_w(['1', '2', '3', '4', '5', '6', '7', '8'], 3)),
            [(['1', '2', '3'], ['2', '3', '4']), (['2', '3', '4'], ['3', '4', '5']), (['3', '4', '5'], ['4', '5', '6']),
             (['4', '5', '6'], ['5', '6', '7']), (['5', '6', '7'], ['6', '7', '8'])])
        self.assertEqual(
            list(shifted_w(['1', '2', '3', '4', '5', '6', '7', '8'], 3)),
            [(['1', '2', '3'], ['2', '3', '4']), (['2', '3', '4'], ['3', '4', '5']), (['3', '4', '5'], ['4', '5', '6']),
             (['4', '5', '6'], ['5', '6', '7']), (['5', '6', '7'], ['6', '7', '8'])])
        self.assertEqual(
            list(shifted_w(['1', '2', '3', '4', '5', '6', '7', '8'], 4)),
            [(['1', '2', '3', '4'], ['2', '3', '4', '5']), (['2', '3', '4', '5'], ['3', '4', '5', '6']),
             (['3', '4', '5', '6'], ['4', '5', '6', '7']), (['4', '5', '6', '7'], ['5', '6', '7', '8'])])
        self.assertEqual(list(shifted_w(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'], 5)),
                         [(['1', '2', '3', '4', '5'], ['2', '3', '4', '5', '6']), (['2', '3', '4', '5', '6'], ['3', '4', '5', '6', '7']),
                          (['3', '4', '5', '6', '7'], ['4', '5', '6', '7', '8']), (['4', '5', '6', '7', '8'], ['5', '6', '7', '8', '9']),
                          (['5', '6', '7', '8', '9'], ['6', '7', '8', '9', '10']), (['6', '7', '8', '9', '10'], ['7', '8', '9', '10', '11']),
                          (['7', '8', '9', '10', '11'], ['8', '9', '10', '11', '12']), (['8', '9', '10', '11', '12'], ['9', '10', '11', '12', '13'])])

    def test_gen_no_overlap_words(self):
        no_overlap_gen = rnnlm.utils.tf_io.generator.gen_no_overlap_words
        self.assertEqual(list(no_overlap_gen([], 2)), [])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3', '4'], 2)), [(['1', '2'], ['2', '3'])])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3', '4'], 3)), [(['1', '2', '3'], ['2', '3', '4'])])
        self.assertEqual(list(no_overlap_gen(['2'], 1)), [])
        self.assertEqual(list(no_overlap_gen(['1'], 2)), [])
        self.assertEqual(list(no_overlap_gen(['1', '2'], 0)), [])
        self.assertEqual(list(no_overlap_gen(['1', '2'], 1)), [(['1'], ['2'])])
        self.assertEqual(list(no_overlap_gen(['1', '2'], 2)), [])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3'], 1)), [(['1'], ['2']), (['2'], ['3'])])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3'], 2)), [(['1', '2'], ['2', '3'])])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3'], 3)), [])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3', '4', '5'], 3)), [(['1', '2', '3'], ['2', '3', '4'])])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3', '4', '5', '6'], 3)),
                         [(['1', '2', '3'], ['2', '3', '4'])])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3', '4', '5'], 2)),
                         [(['1', '2'], ['2', '3']), (['3', '4'], ['4', '5'])])
        self.assertEqual(
            list(no_overlap_gen(['1', '2', '3', '4', '5', '6', '7', '8'], 3)),
            [(['1', '2', '3'], ['2', '3', '4']), (['4', '5', '6'], ['5', '6', '7'])])
        self.assertEqual(
            list(no_overlap_gen(['1', '2', '3', '4', '5', '6', '7', '8'], 4)),
            [(['1', '2', '3', '4'], ['2', '3', '4', '5'])])
        self.assertEqual(
            list(no_overlap_gen(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], 3)),
            [(['1', '2', '3'], ['2', '3', '4']), (['4', '5', '6'], ['5', '6', '7']),
             (['7', '8', '9'], ['8', '9', '10'])])
        self.assertEqual(
            list(no_overlap_gen(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], 4)),
            [(['1', '2', '3', '4'], ['2', '3', '4', '5']), (['5', '6', '7', '8'], ['6', '7', '8', '9'])])
        self.assertEqual(list(no_overlap_gen(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'], 5)),
                         [(['1', '2', '3', '4', '5'], ['2', '3', '4', '5', '6']),
                          (['6', '7', '8', '9', '10'], ['7', '8', '9', '10', '11'])])

        with open('test_data', 'r') as fin, open('output', 'w') as fout:
            res = list(no_overlap_gen(fin, 3))
            fout.write(str(res))

