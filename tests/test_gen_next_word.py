import unittest
from rnnlm.utils.tf_io.extractor import _gen_next_word


class TestGenNextWord(unittest.TestCase):

    def test_empty_file(self):
        with open('test_files/empty') as f:
            g = _gen_next_word(opened_file=f)
            res = list(g)
        self.assertEqual(res, [])

    def test_one_line_file(self):
        with open('test_files/one_line') as f:
            g = _gen_next_word(opened_file=f)
            res = list(g)
        self.assertEqual(res,
                         ['WONDER', 'HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE', 'STUFF',
                          'AT', 'THE', 'MEETINGS', '</s>']
                         )

    def test_short_lines_file(self):
        with open('test_files/short_lines') as f:
            g = _gen_next_word(opened_file=f)
            res = list(g)
        self.assertEqual(res,
                         ['WONDER', 'HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE', 'STUFF',
                          'AT', 'THE', 'MEETINGS', '</s>', 'YEAH', '</s>', 'NOT', 'A', 'LOT', '</s>', 'NO', '</s>',
                          'HMM', '</s>', 'OKAY', '</s>', 'SOUNDS', 'LIKE', "YOU'VE", 'DONE', 'SOME', 'STUFF', '</s>']
                         )
