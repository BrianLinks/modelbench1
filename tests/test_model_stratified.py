import random
import unittest
from collections import Counter

from modelbench.errors import ModelError, SeedError
from modelbench.models import create_model
from modelbench.models.stratified import StratifiedClassifier
from modelbench.seeding import derive_seed

# 75% "a", 25% "b"
X_TRAIN = [[float(i)] for i in range(8)]
Y_TRAIN = ["a", "a", "a", "a", "a", "a", "b", "b"]


def fitted(seed=0):
    model = StratifiedClassifier(seed=seed)
    model.fit(X_TRAIN, Y_TRAIN)
    return model


class StratifiedTests(unittest.TestCase):
    def test_probabilities_are_the_training_priors(self):
        self.assertEqual(fitted().predict_proba([[0.0]]), [[0.75, 0.25]])

    def test_same_seed_gives_identical_predictions_every_time(self):
        X = [[float(i)] for i in range(200)]
        model = fitted(seed=7)
        first = model.predict(X)
        self.assertEqual(model.predict(X), first)
        self.assertEqual(fitted(seed=7).predict(X), first)

    def test_different_seeds_give_different_predictions(self):
        X = [[float(i)] for i in range(200)]
        outputs = {tuple(fitted(seed=s).predict(X)) for s in range(5)}
        self.assertEqual(len(outputs), 5)

    def test_a_row_does_not_depend_on_how_many_rows_follow_it(self):
        model = fitted(seed=3)
        long_run = model.predict([[0.0]] * 50)
        self.assertEqual(model.predict([[0.0]] * 10), long_run[:10])

    def test_feature_values_do_not_matter(self):
        model = fitted(seed=3)
        self.assertEqual(model.predict([[0.0]] * 30), model.predict([[99.0]] * 30))

    def test_long_run_frequencies_match_the_priors(self):
        predictions = fitted(seed=11).predict([[0.0]] * 4000)
        share_a = Counter(predictions)["a"] / 4000
        self.assertAlmostEqual(share_a, 0.75, delta=0.03)

    def test_matches_a_hand_computed_draw_sequence(self):
        # Draws come from random.Random(derive_seed(5, "stratified", "predict")). Recompute the
        # expected labels from the raw draws: draw < 0.75 means "a", otherwise "b".
        draws = random.Random(derive_seed(5, "stratified", "predict"))
        expected = ["a" if draws.random() < 0.75 else "b" for _ in range(40)]
        self.assertEqual(fitted(seed=5).predict([[0.0]] * 40), expected)

    def test_does_not_touch_global_random_state(self):
        random.seed(123)
        expected = random.random()
        random.seed(123)
        fitted(seed=1).predict([[0.0]] * 25)
        self.assertEqual(random.random(), expected)

    def test_single_class_training_data_always_predicts_it(self):
        model = StratifiedClassifier(seed=2)
        model.fit([[0.0], [1.0]], ["only", "only"])
        self.assertEqual(model.predict([[0.0]] * 20), ["only"] * 20)

    def test_every_class_can_be_predicted(self):
        model = StratifiedClassifier(seed=4)
        model.fit([[float(i)] for i in range(9)], ["a", "b", "c"] * 3)
        self.assertEqual(set(model.predict([[0.0]] * 300)), {"a", "b", "c"})

    def test_seed_is_validated_and_exposed_in_params(self):
        for bad in (-1, 2**32, 1.5, "1", True, None):
            with self.assertRaises(SeedError, msg=repr(bad)):
                StratifiedClassifier(seed=bad)
        self.assertEqual(StratifiedClassifier(seed=9).get_params(), {"seed": 9})
        self.assertEqual(create_model("stratified", seed=9).seed, 9)

    def test_unfitted_and_bad_input_are_rejected(self):
        with self.assertRaises(ModelError):
            StratifiedClassifier().predict([[0.0]])
        with self.assertRaises(ModelError):
            fitted().predict([[1.0, 2.0]])
        with self.assertRaises(ModelError):
            fitted().predict([])


if __name__ == "__main__":
    unittest.main()
