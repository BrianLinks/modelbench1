import unittest

from modelbench.errors import ModelError
from modelbench.models import REGISTRY, available_models, create_model
from modelbench.models.base import check_features, check_labels, class_priors
from modelbench.models.majority import MajorityClassifier


class CheckFeaturesTests(unittest.TestCase):
    def test_returns_floats_as_lists(self):
        self.assertEqual(check_features([[1, 2], (3, 4.5)]), [[1.0, 2.0], [3.0, 4.5]])

    def test_rejects_bad_shapes_and_values(self):
        bad = [
            [],
            "abc",
            b"ab",
            None,
            [[]],
            [[1, 2], [3]],
            ["ab"],
            [1, 2],
            [[1, "x"]],
            [[1, None]],
            [[True]],
            [[float("nan")]],
            [[float("inf")]],
            [[1, 2], [3, float("-inf")]],
        ]
        for value in bad:
            with self.assertRaises(ModelError, msg=repr(value)):
                check_features(value)

    def test_error_points_at_the_offending_cell(self):
        with self.assertRaises(ModelError) as ctx:
            check_features([[1, 2], [3, "x"]], where="X_test")
        self.assertIn("X_test[1][1]", str(ctx.exception))


class CheckLabelsTests(unittest.TestCase):
    def test_accepts_hashable_labels(self):
        self.assertEqual(check_labels(["a", 1, (2, 3)], 3), ["a", 1, (2, 3)])

    def test_rejects_length_mismatch_none_and_unhashable(self):
        with self.assertRaises(ModelError):
            check_labels(["a"], 2)
        with self.assertRaises(ModelError):
            check_labels(["a", None], 2)
        with self.assertRaises(ModelError):
            check_labels(["a", ["b"]], 2)
        with self.assertRaises(ModelError):
            check_labels("ab", 2)


class ClassPriorTests(unittest.TestCase):
    def test_priors_follow_class_order(self):
        self.assertEqual(class_priors(["a", "b", "a", "a"], ["a", "b", "c"]), [0.75, 0.25, 0.0])


class RegistryTests(unittest.TestCase):
    def test_lookup_and_listing(self):
        self.assertIsInstance(create_model("majority"), MajorityClassifier)
        self.assertEqual(available_models(), tuple(sorted(REGISTRY)))
        self.assertIn("majority", available_models())

    def test_unknown_model_or_parameter(self):
        with self.assertRaises(ModelError):
            create_model("no-such-model")
        with self.assertRaises(ModelError):
            create_model(["unhashable"])
        with self.assertRaises(ModelError):
            create_model("majority", k=3)

    def test_every_registered_model_follows_the_interface_contract(self):
        X = [[float(i)] for i in range(8)]
        y = ["a", "a", "a", "a", "b", "b", "b", "b"]
        for name in available_models():
            model = create_model(name)
            with self.assertRaises(ModelError, msg=name):  # not fitted
                model.predict(X)
            with self.assertRaises(ModelError, msg=name):
                model.classes_
            model.fit(X, y)
            self.assertEqual(model.classes_, ("a", "b"), name)
            self.assertEqual(len(model.predict(X)), 8, name)
            for row in model.predict_proba(X):
                self.assertEqual(len(row), 2, name)
                self.assertAlmostEqual(sum(row), 1.0, msg=name)
                self.assertTrue(all(0.0 <= p <= 1.0 for p in row), name)
            with self.assertRaises(ModelError, msg=name):  # wrong feature count
                model.predict([[1.0, 2.0]])
            rebuilt = create_model(name, **model.get_params())
            self.assertEqual(rebuilt.get_params(), model.get_params(), name)

    def test_float_overflow_surfaces_as_model_error_for_every_model(self):
        extreme = [[1e300], [-1e300]] * 4
        labels = ["a", "b"] * 4
        sane = [[float(i)] for i in range(8)]
        for name in available_models():
            model = create_model(name)
            try:  # a model may cope with extreme values; if it cannot, it must say so cleanly
                model.fit(extreme, labels)
                model.predict_proba([[1e300], [-1e300]])
            except ModelError:
                pass
            model = create_model(name)
            model.fit(sane, labels)
            try:
                model.predict_proba([[1e300], [-1e300]])
                model.predict([[1e300]])
            except ModelError:
                pass

    def test_overflow_in_training_is_reported_and_leaves_the_model_unfitted(self):
        model = create_model("gaussian_nb")
        with self.assertRaisesRegex(ModelError, "overflow"):
            model.fit([[1e300], [-1e300]] * 2, ["a", "b"] * 2)
        with self.assertRaises(ModelError):
            model.classes_
        model.fit([[1.0], [2.0]], ["a", "b"])
        with self.assertRaisesRegex(ModelError, "overflow"):
            model.predict([[1e300]])
        with self.assertRaisesRegex(ModelError, "overflow"):
            model.predict_proba([[1e300]])

    def test_invalid_data_in_a_refit_keeps_the_previous_fit(self):
        model = create_model("majority")
        model.fit([[0.0], [1.0]], ["a", "b"])
        with self.assertRaises(ModelError):
            model.fit([[0.0], [float("nan")]], ["a", "b"])
        # validation failed before training started, so nothing was replaced
        self.assertEqual(model.classes_, ("a", "b"))

    def test_a_crash_during_training_leaves_the_model_unfitted(self):
        class Exploding(MajorityClassifier):
            fail = False

            def _fit(self, X, y):
                if self.fail:
                    raise RuntimeError("training blew up")
                super()._fit(X, y)

        model = Exploding()
        model.fit([[0.0], [1.0]], ["a", "b"])
        model.fail = True
        with self.assertRaises(RuntimeError):
            model.fit([[0.0], [1.0]], ["x", "y"])
        with self.assertRaises(ModelError):  # no half-trained model is left behind
            model.predict([[0.0]])


if __name__ == "__main__":
    unittest.main()
