import unittest

from rnnlm.utils.global_var_manager import GlobalVarManager


class TestGlobalVarManager(unittest.TestCase):

    def setUp(self):
        self.l1 = []
        self.l2 = [1, 2]
        self.d1 = dict()
        self.d2 = {"a": 1, "b": 2}
        self.s1 = set()
        self.s2 = {"2"}
        self.b1 = True
        self.b2 = False
        self.bl1 = [True, False]
        self.dl1 = {"a": True, "b": False}
        self.i1 = 0
        self.i2 = 15.3
        self.i3 = 10
        self.st1 = ""
        self.st2 = "abc123"
        self.st3 = "True"

        self.all = [self.l1, self.l2, self.d1, self.d2, self.s1,
                    self.s2, self.b1, self.b2, self.bl1, self.dl1,
                    self.i1, self.i2, self.i3, self.st1, self.st2,
                    self.st3]
        self.all_d = {str(i): v for i, v in enumerate(self.all)}

    def test_successful_creation(self):
        gm = GlobalVarManager(self.all_d)
        self.assertEqual(list(gm.vars.values()), self.all)
        self.assertEqual(gm.initial_values, gm.vars)

    def test_add_var_after_creation(self):
        gm = GlobalVarManager({"1": self.st1, "2": self.dl1})
        gm.add({"3": self.st2})
        self.assertEqual(list(gm.vars.values()), [self.st1, self.dl1, self.st2])
        self.assertEqual(list(gm.initial_values.values()), [self.st1, self.dl1, self.st2])

    def test_add_var_that_already_exists_should_throw_error(self):
        gm = GlobalVarManager(self.all_d)

        for i, test_element in self.all_d.items():

            with self.assertRaises(ValueError) as e:
                gm.add({i: test_element})

            self.assertRegex(str(e.exception), "^GlobalVarManager has already var.*")

    def test_update_var(self):
        gm = GlobalVarManager({"1": self.st2})
        gm.update({"1": "hello"})
        self.assertEqual(list(gm.vars.values()), ["hello"])

    def test_update_var_that_does_not_exists_should_throw_error(self):
        gm = GlobalVarManager()
        with self.assertRaises(ValueError) as e:
            gm.update({"1": "hello"})

        self.assertRegex(str(e.exception), "^Can't update var .* which does not exist. Not updating anything$")

    def test_retrieve_original_value_after_it_has_changed(self):
        gm = GlobalVarManager({"1": self.st2})
        gm.update({"1": "hello"})
        self.assertEqual([self.st2], list(gm.initial_values.values()))

    def test_remove_var(self):
        gm = GlobalVarManager({"1": self.st2})
        gm.remove(["1"])

        self.assertEqual(list(gm.vars), [])
        self.assertEqual(list(gm.initial_values), [])
        self.assertNotIn(self.st2, list(gm.vars))

    def test_remove_var_that_does_not_exist_should_throw_error(self):
        gm = GlobalVarManager({"1": self.st1})

        with self.assertRaises(ValueError) as e:
            gm.remove(["2"])

        self.assertRegex(str(e.exception), "^Can't remove var .* which does not exists$")

    def test_reset_all(self):
        gm = GlobalVarManager(self.all_d)

        updated_values = [["sdf"], [1], self.d2, self.d1, {"123"}, {"21"}, True, True,
                          [False, False, True], {"a": True, "b": False}, -1, "15.3",
                          0, "None", "abc123", "True"]

        update_dict = dict(zip(self.all_d.keys(), updated_values))
        gm.update(update_dict)

        self.assertEqual(list(gm.vars.values()), updated_values)
        self.assertEqual(list(gm.initial_values.values()), self.all)
        gm.reset_all()
        self.assertEqual(list(gm.vars.values()), self.all)

    def test_reset_all_when_no_vars(self):
        gm = GlobalVarManager()
        gm.reset_all()

    def test_reset_one(self):
        gm = GlobalVarManager({"1": self.i1})
        gm.update({"1": 999})
        self.assertEqual(list(gm.vars.values()), [999])
        gm.reset_one("1")

        self.assertEqual(list(gm.vars.values()), [self.i1])

    def test_reset_one_var_does_not_exist_should_throw_error(self):
        gm = GlobalVarManager({"1": self.i1})
        gm.update({"1": 999})
        self.assertEqual(list(gm.vars.values()), [999])
        with self.assertRaises(ValueError) as e:
            gm.reset_one("2")

        self.assertRegex(str(e.exception), "^Cannot reset var .* since it does not exist$")
