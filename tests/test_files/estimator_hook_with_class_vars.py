import tensorflow as tf


class HookWithClassVars(tf.train.SessionRunHook):

    a = 1
    b = 2
    c = 3
    _hidden = 4

    def __int__(self):
        self.l1 = 10
        self.l2 = 11
        self.l3 = 12

    def _private_method(self):
        l5 = self.l1
        return l5

    def public_method(self):
        l6 = HookWithClassVars.a

        return l6
