import unittest

from rnnlm.utils.tf_io.extractor import extract_x_without_overlap_y_shifted_by_1


class TestGenNoOverlap(unittest.TestCase):

    def test_empty_file_with_seq_len(self):
        with open('test_files/empty') as f:
            g = extract_x_without_overlap_y_shifted_by_1(file_obj=f, seq_len=2)
            res = list(g)
        self.assertEqual(len(res), 0)

    def test_empty_file_without_seq_len(self):
        with open('test_files/empty') as f:
            g = extract_x_without_overlap_y_shifted_by_1(file_obj=f, seq_len=0)
            res = list(g)
        self.assertEqual(res, [])

    def test_one_line_with_leftover(self):
        with open('test_files/one_line') as f:
            g = extract_x_without_overlap_y_shifted_by_1(file_obj=f, seq_len=4)
            res = list(g)
        self.assertEqual(res,
                         [
                             (['WONDER', 'HOW', 'MUCH', 'OF'], ['HOW', 'MUCH', 'OF', 'THE']),
                             (['THE', 'MEETINGS', 'IS', 'TALKING'], ['MEETINGS', 'IS', 'TALKING', 'ABOUT']),
                             (['ABOUT', 'THE', 'STUFF', 'AT'], ['THE', 'STUFF', 'AT', 'THE'])
                         ]
                         )

    def test_one_line_without_leftover(self):
        with open('test_files/one_line') as f:
            g = extract_x_without_overlap_y_shifted_by_1(file_obj=f, seq_len=7)
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
                             )
                         ]
                         )

    def test_one_line_with_seq_len_of_1(self):
        with open('test_files/one_line') as f:
            g = extract_x_without_overlap_y_shifted_by_1(file_obj=f, seq_len=1)
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
            g = extract_x_without_overlap_y_shifted_by_1(file_obj=f, seq_len=15)
            res = list(g)
        self.assertEqual(res, [])

    def test_one_line_seq_len_bigger_than_words_in_file(self):
        with open('test_files/one_line') as f:
            g = extract_x_without_overlap_y_shifted_by_1(file_obj=f, seq_len=16)
            res = list(g)
        self.assertEqual(res, [])

    def test_short_lines_with_leftover(self):
        with open('test_files/short_lines') as f:
            g = extract_x_without_overlap_y_shifted_by_1(file_obj=f, seq_len=7)
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

    def test_short_lines_without_leftover(self):
        with open('test_files/short_lines') as f:
            g = extract_x_without_overlap_y_shifted_by_1(file_obj=f, seq_len=3)
            res = list(g)
        self.assertEqual(res,
                         [
                             (['WONDER', 'HOW', 'MUCH'], ['HOW', 'MUCH', 'OF']),
                             (['OF', 'THE', 'MEETINGS'], ['THE', 'MEETINGS', 'IS']),
                             (['IS', 'TALKING', 'ABOUT'], ['TALKING', 'ABOUT', 'THE']),
                             (['THE', 'STUFF', 'AT'], ['STUFF', 'AT', 'THE']),
                             (['THE', 'MEETINGS', '</s>'], ['MEETINGS', '</s>', 'YEAH']),
                             (['YEAH', '</s>', 'NOT'], ['</s>', 'NOT', 'A']),
                             (['A', 'LOT', '</s>'], ['LOT', '</s>', 'NO']),
                             (['NO', '</s>', 'HMM'], ['</s>', 'HMM', '</s>']),
                             (['</s>', 'OKAY', '</s>'], ['OKAY', '</s>', 'SOUNDS']),
                             (['SOUNDS', 'LIKE', "YOU'VE"], ['LIKE', "YOU'VE", 'DONE']),
                             (['DONE', 'SOME', 'STUFF'], ['SOME', 'STUFF', '</s>'])
                         ]
                         )

    def test_short_lines_with_seq_len_that_y_overlaps_lines(self):
        with open('test_files/short_lines') as f:
            g = extract_x_without_overlap_y_shifted_by_1(file_obj=f, seq_len=15)
            res = list(g)
        self.assertEqual(res,
                         [
                             (['WONDER', 'HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE',
                               'STUFF', 'AT', 'THE', 'MEETINGS', '</s>'],
                              ['HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE', 'STUFF', 'AT',
                               'THE', 'MEETINGS', '</s>', 'YEAH']),
                             (['YEAH', '</s>', 'NOT', 'A', 'LOT', '</s>', 'NO', '</s>', 'HMM', '</s>', 'OKAY', '</s>',
                               'SOUNDS', 'LIKE', "YOU'VE"],
                              ['</s>', 'NOT', 'A', 'LOT', '</s>', 'NO', '</s>', 'HMM', '</s>', 'OKAY', '</s>', 'SOUNDS',
                               'LIKE', "YOU'VE", 'DONE'])
                         ]
                         )

    def test_short_lines_with_seq_len_that_x_overlaps_lines(self):
        with open('test_files/short_lines') as f:
            g = extract_x_without_overlap_y_shifted_by_1(file_obj=f, seq_len=16)
            res = list(g)
        self.assertEqual(res,
                         [
                             (['WONDER', 'HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE',
                               'STUFF', 'AT', 'THE', 'MEETINGS', '</s>', 'YEAH'],
                              ['HOW', 'MUCH', 'OF', 'THE', 'MEETINGS', 'IS', 'TALKING', 'ABOUT', 'THE', 'STUFF', 'AT',
                               'THE', 'MEETINGS', '</s>', 'YEAH', '</s>']),
                             (['</s>', 'NOT', 'A', 'LOT', '</s>', 'NO', '</s>', 'HMM', '</s>', 'OKAY', '</s>',
                               'SOUNDS', 'LIKE', "YOU'VE", 'DONE', 'SOME'],
                              ['NOT', 'A', 'LOT', '</s>', 'NO', '</s>', 'HMM', '</s>', 'OKAY', '</s>', 'SOUNDS',
                               'LIKE', "YOU'VE", 'DONE', 'SOME', 'STUFF'])
                         ]
                         )
