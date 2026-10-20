import unittest

from modelbench.errors import ModelError
from modelbench.models.majority import MajorityClassifier


class MajorityTests(unittest.TestCase):
    def test_predicts_the_most_common_class_for_every_row(self):
        model = MajorityClassifier()
        model.fit([[0], [1], [2]], ["a", "b", "b"])
        self.assertEqual(model.predict([[9], [-5], [0]]), ["b", "b", "b"])

    def test_probabilities_are_the_training_priors(self):
        model = MajorityClassifier()
        model.fit([[0], [1], [2], [3]], ["a", "b", "b", "b"])
        self.assertEqual(model.classes_, ("a", "b"))
        self.assertEqual(model.predict_proba([[5], [6]]), [[0.25, 0.75], [0.25, 0.75]])

    def test_ties_go_to_the_first_class_in_sorted_order(self):
        model = MajorityClassifier()
        model.fit([[0], [1]], ["b", "a"])
        self.assertEqual(model.classes_, ("a", "b"))
        self.assertEqual(model.predict([[0]]), ["a"])

    def test_single_class_training_data(self):
        model = MajorityClassifier()
        model.fit([[0], [1]], [7, 7])
        self.assertEqual(model.predict([[3]]), [7])
        self.assertEqual(model.predict_proba([[3]]), [[1.0]])

    def test_integer_and_string_labels(self):
        model = MajorityClassifier()
        model.fit([[0]] * 5, [3, 1, 3, 1, 3])
        self.assertEqual(model.predict([[0]]), [3])

    def test_features_are_ignored_but_still_validated(self):
        model = MajorityClassifier()
        model.fit([[0, 1], [1, 0]], ["a", "a"])
        with self.assertRaises(ModelError):
            model.predict([[1.0]])
        with self.assertRaises(ModelError):
            model.predict([[float("nan"), 1.0]])

    def test_refit_replaces_the_previous_model(self):
        model = MajorityClassifier()
        model.fit([[0], [1]], ["a", "a"])
        model.fit([[0], [1], [2]], ["x", "y", "y"])
        self.assertEqual(model.classes_, ("x", "y"))
        self.assertEqual(model.predict([[0]]), ["y"])

    def test_models_do_not_share_state(self):
        first, second = MajorityClassifier(), MajorityClassifier()
        first.fit([[0]], ["a"])
        second.fit([[0]], ["b"])
        self.assertEqual((first.predict([[0]]), second.predict([[0]])), (["a"], ["b"]))


if __name__ == "__main__":
    unittest.main()
