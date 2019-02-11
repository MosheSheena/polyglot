import unittest

from rnnlm.utils.tf_io.extractor import _gen_n_words


class TestGenNWords(unittest.TestCase):

    def test_empty_file_with_positive_n(self):
        with open('test_files/empty') as f:
            g = _gen_n_words(opened_file=f, n=2)
            res = list(g)
        self.assertEqual(res, [])

    def test_empty_file_with_n_zero(self):
        with open('test_files/empty') as f:
            g = _gen_n_words(opened_file=f, n=0)
            res = list(g)
        self.assertEqual(res, [])

    def test_one_line_file_with_n_1(self):
        with open('test_files/one_line') as f:
            g = _gen_n_words(opened_file=f, n=1)
            res = list(g)
        self.assertEqual(res,
                         [
                             ['WONDER'], ['HOW'], ['MUCH'], ['OF'],
                             ['THE'], ['MEETINGS'], ['IS'], ['TALKING'],
                             ['ABOUT'], ['THE'], ['STUFF'], ['AT'], ['THE'],
                             ['MEETINGS'], ['</s>']
                         ]
                         )

    def test_one_line_file_with_leftover(self):
        with open('test_files/one_line') as f:
            g = _gen_n_words(opened_file=f, n=4)
            res = list(g)
        self.assertEqual(res,
                         [
                             ['WONDER', 'HOW', 'MUCH', 'OF'],
                             ['THE', 'MEETINGS', 'IS', 'TALKING'],
                             ['ABOUT', 'THE', 'STUFF', 'AT']
                         ]
                         )

    def test_one_line_file_without_leftover(self):
        with open('test_files/one_line') as f:
            g = _gen_n_words(opened_file=f, n=5)
            res = list(g)
        self.assertEqual(res,
                         [
                             ['WONDER', 'HOW', 'MUCH', 'OF', 'THE'],
                             ['MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE'],
                             ['STUFF', 'AT', 'THE', 'MEETINGS', '</s>']
                         ]
                         )

    def test_one_line_file_n_less_than_line_len(self):
        with open('test_files/short_lines') as f:
            g = _gen_n_words(opened_file=f, n=4)
            res = list(g)
        self.assertEqual(res,
                         [
                             ['WONDER', 'HOW', 'MUCH', 'OF'],
                             ['THE', 'MEETINGS', 'IS', 'TALKING'],
                             ['ABOUT', 'THE', 'STUFF', 'AT'],
                             ['THE', 'MEETINGS', '</s>', 'YEAH'],
                             ['</s>', 'NOT', 'A', 'LOT'],
                             ['</s>', 'NO', '</s>', 'HMM'],
                             ['</s>', 'OKAY', '</s>', 'SOUNDS'],
                             ['LIKE', "YOU'VE", 'DONE', 'SOME']
                         ]
                         )

    def test_one_line_file_n_equal_line_len(self):
        with open('test_files/short_lines') as f:
            g = _gen_n_words(opened_file=f, n=15)
            res = list(g)
        self.assertEqual(res,
                         [
                             ['WONDER', 'HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE',
                              'STUFF', 'AT', 'THE', 'MEETINGS', '</s>'],
                             ['YEAH', '</s>', 'NOT', 'A', 'LOT', '</s>', 'NO', '</s>', 'HMM', '</s>', 'OKAY', '</s>',
                              'SOUNDS', 'LIKE', "YOU'VE"],
                         ]
                         )

    def test_one_line_file_n_more_than_line_len(self):
        with open('test_files/short_lines') as f:
            g = _gen_n_words(opened_file=f, n=17)
            res = list(g)
        self.assertEqual(res,
                         [
                             ['WONDER', 'HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE',
                              'STUFF', 'AT', 'THE', 'MEETINGS', '</s>', 'YEAH', '</s>'],
                             ['NOT', 'A', 'LOT', '</s>', 'NO', '</s>', 'HMM', '</s>', 'OKAY', '</s>',
                              'SOUNDS', 'LIKE', "YOU'VE", 'DONE', 'SOME', 'STUFF', '</s>'],
                         ]
                         )
