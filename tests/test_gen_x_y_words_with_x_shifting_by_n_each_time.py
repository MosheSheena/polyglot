import unittest
from rnnlm.utils.tf_io.extractor import extract_x_y_words_with_x_shifting_by_n_each_time


class TestExtractXYWordsWithXShiftingByNEachTime(unittest.TestCase):

    def test_empty_file_with_n_0(self):
        with open('test_files/empty') as f:
            g = extract_x_y_words_with_x_shifting_by_n_each_time(file_obj=f, seq_len=2, n=0)
            res = list(g)
        self.assertEqual(len(res), 0)

    def test_empty_file_with_n_1(self):
        with open('test_files/empty') as f:
            g = extract_x_y_words_with_x_shifting_by_n_each_time(file_obj=f, seq_len=2, n=1)
            res = list(g)
        self.assertEqual(res, [])

    def test_one_line_without_leftover(self):
        with open('test_files/one_line') as f:
            g = extract_x_y_words_with_x_shifting_by_n_each_time(file_obj=f, seq_len=4, n=2)
            res = list(g)
        self.assertEqual(res,
                         [
                             (['WONDER', 'HOW', 'MUCH', 'OF'], ['HOW', 'MUCH', 'OF', 'THE']),
                             (['MUCH', 'OF', 'THE', 'MEETINGS'], ['OF', 'THE', 'MEETINGS', 'IS']),
                             (['THE', 'MEETINGS', 'IS', 'TALKING'], ['MEETINGS', 'IS', 'TALKING', 'ABOUT']),
                             (['IS', 'TALKING', 'ABOUT', 'THE'], ['TALKING', 'ABOUT', 'THE', 'STUFF']),
                             (['ABOUT', 'THE', 'STUFF', 'AT'], ['THE', 'STUFF', 'AT', 'THE']),
                             (['STUFF', 'AT', 'THE', 'MEETINGS'], ['AT', 'THE', 'MEETINGS', '</s>'])
                         ]
                         )

    def test_one_line_with_leftover(self):
        with open('test_files/one_line') as f:
            g = extract_x_y_words_with_x_shifting_by_n_each_time(file_obj=f, seq_len=7, n=5)
            res = list(g)
        self.assertEqual(res,
                         [
                             (
                                 ['WONDER', 'HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS'],
                                 ['HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING']
                             ),
                             (
                                 ['MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE', 'STUFF', 'AT'],
                                 ['IS', 'TALKING', 'ABOUT', 'THE', 'STUFF', 'AT', 'THE']
                             )
                         ]
                         )

    def test_one_line_with_seq_len_of_1(self):
        with open('test_files/one_line') as f:
            g = extract_x_y_words_with_x_shifting_by_n_each_time(file_obj=f, seq_len=1, n=1)
            res = list(g)
        self.assertEqual(res,
                         [
                             (['WONDER'], ['HOW']), (['HOW'], ['MUCH']),
                             (['MUCH'], ['OF']), (['OF'], ['THE']),
                             (['THE'], ['MEETINGS']), (['MEETINGS'], ['IS']),
                             (['IS'], ['TALKING']), (['TALKING'], ['ABOUT']),
                             (['ABOUT'], ['THE']), (['THE'], ['STUFF']),
                             (['STUFF'], ['AT']), (['AT'], ['THE']),
                             (['THE'], ['MEETINGS']), (['MEETINGS'], ['</s>'])
                         ]
                         )

    def test_one_line_seq_len_equal_num_words_in_file(self):
        with open('test_files/one_line') as f:
            g = extract_x_y_words_with_x_shifting_by_n_each_time(file_obj=f, seq_len=15, n=15)
            res = list(g)
        self.assertEqual(res, [])

    def test_one_line_seq_len_bigger_than_words_in_file(self):
        with open('test_files/one_line') as f:
            g = extract_x_y_words_with_x_shifting_by_n_each_time(file_obj=f, seq_len=16, n=16)
            res = list(g)
        self.assertEqual(res, [])

    def test_one_line_n_bigger_than_seq_len(self):
        with open('test_files/one_line') as f:
            g = extract_x_y_words_with_x_shifting_by_n_each_time(file_obj=f, seq_len=4, n=5)

        self.assertRaisesRegex(AssertionError, '.*cannot overlap more than sequence length.*',
                               lambda: list(g))

    def test_short_lines_with_leftover(self):
        with open('test_files/short_lines') as f:
            g = extract_x_y_words_with_x_shifting_by_n_each_time(file_obj=f, seq_len=7, n=6)
            res = list(g)
        self.assertEqual(res,
                         [
                             (
                                 ['WONDER', 'HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS'],
                                 ['HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING']
                             ),
                             (
                                 ['IS', 'TALKING', 'ABOUT', 'THE', 'STUFF', 'AT', 'THE'],
                                 ['TALKING', 'ABOUT', 'THE', 'STUFF', 'AT', 'THE', 'MEETINGS']
                             ),
                             (
                                 ['THE', 'MEETINGS', '</s>', 'YEAH', '</s>', 'NOT', 'A'],
                                 ['MEETINGS', '</s>', 'YEAH', '</s>', 'NOT', 'A', 'LOT']
                             ),
                             (
                                 ['A', 'LOT', '</s>', 'NO', '</s>', 'HMM', '</s>'],
                                 ['LOT', '</s>', 'NO', '</s>', 'HMM', '</s>', 'OKAY']
                             ),
                             (
                                 ['</s>', 'OKAY', '</s>', 'SOUNDS', 'LIKE', "YOU'VE", 'DONE'],
                                 ['OKAY', '</s>', 'SOUNDS', 'LIKE', "YOU'VE", 'DONE', 'SOME']
                             )
                         ]
                         )

    def test_short_lines_with_n_equal_to_seq_len(self):
        with open('test_files/short_lines') as f:
            g = extract_x_y_words_with_x_shifting_by_n_each_time(file_obj=f, seq_len=7, n=7)
            res = list(g)
        self.assertEqual(res,
                         [
                             (
                                 ['WONDER', 'HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS'],
                                 ['HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING']
                             ),
                             (
                                 ['TALKING', 'ABOUT', 'THE', 'STUFF', 'AT', 'THE', 'MEETINGS'],
                                 ['ABOUT', 'THE', 'STUFF', 'AT', 'THE', 'MEETINGS', '</s>']
                             ),
                             (
                                 ['</s>', 'YEAH', '</s>', 'NOT', 'A', 'LOT', '</s>'],
                                 ['YEAH', '</s>', 'NOT', 'A', 'LOT', '</s>', 'NO']
                             ),
                             (
                                 ['NO', '</s>', 'HMM', '</s>', 'OKAY', '</s>', 'SOUNDS'],
                                 ['</s>', 'HMM', '</s>', 'OKAY', '</s>', 'SOUNDS', 'LIKE']
                             )
                         ]
                         )

    def test_short_lines_n_on_short_lines(self):
        with open('test_files/short_lines') as f:
            g = extract_x_y_words_with_x_shifting_by_n_each_time(file_obj=f, seq_len=8, n=4)
            res = list(g)
        self.assertEqual(res,
                         [
                             (
                                 ['WONDER', 'HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING'],
                                 ['HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING', 'ABOUT']
                             ),
                             (
                                 ['THE', 'MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE', 'STUFF', 'AT'],
                                 ['MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE', 'STUFF', 'AT', 'THE']
                             ),
                             (
                                 ['ABOUT', 'THE', 'STUFF', 'AT', 'THE', 'MEETINGS', '</s>', 'YEAH'],
                                 ['THE', 'STUFF', 'AT', 'THE', 'MEETINGS', '</s>', 'YEAH', '</s>']
                             ),
                             (
                                 ['THE', 'MEETINGS', '</s>', 'YEAH', '</s>', 'NOT', 'A', 'LOT'],
                                 ['MEETINGS', '</s>', 'YEAH', '</s>', 'NOT', 'A', 'LOT', '</s>']
                             ),
                             (
                                 ['</s>', 'NOT', 'A', 'LOT', '</s>', 'NO', '</s>', 'HMM'],
                                 ['NOT', 'A', 'LOT', '</s>', 'NO', '</s>', 'HMM', '</s>']
                             ),
                             (
                                 ['</s>', 'NO', '</s>', 'HMM', '</s>', 'OKAY', '</s>', 'SOUNDS'],
                                 ['NO', '</s>', 'HMM', '</s>', 'OKAY', '</s>', 'SOUNDS', 'LIKE']
                             ),
                             (
                                 ['</s>', 'OKAY', '</s>', 'SOUNDS', 'LIKE', "YOU'VE", 'DONE', 'SOME'],
                                 ['OKAY', '</s>', 'SOUNDS', 'LIKE', "YOU'VE", 'DONE', 'SOME', 'STUFF']
                             )
                         ]
                         )

    def test_short_lines_with_seq_len_and_n_that_overlaps_lines(self):
        with open('test_files/short_lines') as f:
            g = extract_x_y_words_with_x_shifting_by_n_each_time(file_obj=f, seq_len=17, n=16)
            res = list(g)
        self.assertEqual(res,
                         [
                             (
                                 ['WONDER', 'HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE',
                                  'STUFF', 'AT', 'THE', 'MEETINGS', '</s>', 'YEAH', '</s>'],
                                 ['HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE', 'STUFF',
                                  'AT', 'THE', 'MEETINGS', '</s>', 'YEAH', '</s>', 'NOT']
                             ),
                             (
                                 ['</s>', 'NOT', 'A', 'LOT', '</s>', 'NO', '</s>', 'HMM', '</s>', 'OKAY', '</s>',
                                  'SOUNDS', 'LIKE', "YOU'VE", 'DONE', 'SOME', 'STUFF'],
                                 ['NOT', 'A', 'LOT', '</s>', 'NO', '</s>', 'HMM', '</s>', 'OKAY', '</s>',
                                  'SOUNDS', 'LIKE', "YOU'VE", 'DONE', 'SOME', 'STUFF', '</s>']
                             )
                         ]
                         )
