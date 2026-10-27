import math
import unittest

from modelbench.errors import StatsError
from modelbench.stats import (
    mcnemar_exact,
    paired_t_test,
    regularized_incomplete_beta,
    student_t_two_sided_p,
)


class IncompleteBetaTests(unittest.TestCase):
    def test_known_closed_forms(self):
        # I_x(1, 1) = x and I_x(2, 1) = x^2 and I_x(1, 2) = 1 - (1 - x)^2
        for x in (0.1, 0.5, 0.9):
            self.assertAlmostEqual(regularized_incomplete_beta(1, 1, x), x, places=12)
            self.assertAlmostEqual(regularized_incomplete_beta(2, 1, x), x * x, places=12)
            self.assertAlmostEqual(
                regularized_incomplete_beta(1, 2, x), 1 - (1 - x) ** 2, places=12
            )
        self.assertAlmostEqual(regularized_incomplete_beta(3, 3, 0.5), 0.5, places=12)

    def test_edges_and_validation(self):
        self.assertEqual(regularized_incomplete_beta(2, 3, 0.0), 0.0)
        self.assertEqual(regularized_incomplete_beta(2, 3, 1.0), 1.0)
        for a, b, x in ((0, 1, 0.5), (1, -1, 0.5), (1, 1, -0.1), (1, 1, 1.1)):
            with self.subTest((a, b, x)), self.assertRaises(StatsError):
                regularized_incomplete_beta(a, b, x)

    def test_symmetry(self):
        for a, b, x in ((2.5, 4.0, 0.3), (0.5, 7.0, 0.8), (10.0, 0.5, 0.95)):
            left = regularized_incomplete_beta(a, b, x)
            right = 1.0 - regularized_incomplete_beta(b, a, 1.0 - x)
            self.assertAlmostEqual(left, right, places=10)


class StudentTTests(unittest.TestCase):
    def test_closed_forms_for_one_and_two_degrees_of_freedom(self):
        # df = 1 is the Cauchy distribution: p = 1 - (2 / pi) * atan(|t|)
        for t in (0.5, 1.0, 3.0):
            self.assertAlmostEqual(
                student_t_two_sided_p(t, 1), 1 - 2 / math.pi * math.atan(t), places=10
            )
        # df = 2: p = 1 - |t| / sqrt(2 + t^2)
        for t in (0.5, 2.0, 5.0):
            self.assertAlmostEqual(
                student_t_two_sided_p(t, 2), 1 - t / math.sqrt(2 + t * t), places=10
            )

    def test_published_critical_values(self):
        # two-sided 5% critical values of Student's t
        for df, t in ((4, 2.776445), (10, 2.228139), (30, 2.042272)):
            self.assertAlmostEqual(student_t_two_sided_p(t, df), 0.05, places=5)
        self.assertAlmostEqual(student_t_two_sided_p(63.656741, 1), 0.01, places=5)

    def test_symmetric_and_edge_cases(self):
        self.assertEqual(student_t_two_sided_p(0.0, 5), 1.0)
        self.assertAlmostEqual(student_t_two_sided_p(-2.0, 7), student_t_two_sided_p(2.0, 7))
        self.assertEqual(student_t_two_sided_p(math.inf, 5), 0.0)
        self.assertLess(student_t_two_sided_p(1e6, 5), 1e-12)
        with self.assertRaises(StatsError):
            student_t_two_sided_p(1.0, 0)
        with self.assertRaises(StatsError):
            student_t_two_sided_p(float("nan"), 3)

    def test_a_large_df_approaches_the_normal_distribution(self):
        # P(|Z| >= 1.96) = 0.05
        self.assertAlmostEqual(student_t_two_sided_p(1.959964, 100000), 0.05, places=4)


class PairedTTestTests(unittest.TestCase):
    def test_hand_computed_example(self):
        # differences 1, 2, 3, 4 (a - b): mean 2.5, sample variance 5/3, se = sqrt(5/12)
        a = [2.0, 4.0, 6.0, 8.0]
        b = [1.0, 2.0, 3.0, 4.0]
        result = paired_t_test(a, b)
        self.assertEqual(result.n, 4)
        self.assertEqual(result.df, 3)
        self.assertAlmostEqual(result.mean_difference, 2.5)
        self.assertAlmostEqual(result.std_error, math.sqrt(5 / 12))
        self.assertAlmostEqual(result.t, 2.5 / math.sqrt(5 / 12))
        self.assertAlmostEqual(result.p_value, student_t_two_sided_p(result.t, 3))
        self.assertTrue(0.02 < result.p_value < 0.06)  # t = 3.873 on 3 df: p about 0.03

    def test_two_pairs_matches_the_cauchy_closed_form(self):
        # differences 1 and 3: mean 2, sample sd sqrt(2), se = 1, t = 2 on 1 df
        result = paired_t_test([2.0, 4.0], [1.0, 1.0])
        self.assertAlmostEqual(result.t, 2.0)
        self.assertAlmostEqual(result.p_value, 1 - 2 / math.pi * math.atan(2.0), places=10)

    def test_the_sign_follows_a_minus_b(self):
        up = paired_t_test([3.0, 4.0, 5.0, 7.0], [1.0, 2.0, 2.0, 3.0])
        down = paired_t_test([1.0, 2.0, 2.0, 3.0], [3.0, 4.0, 5.0, 7.0])
        self.assertGreater(up.t, 0)
        self.assertAlmostEqual(down.t, -up.t)
        self.assertAlmostEqual(down.p_value, up.p_value)

    def test_degenerate_differences(self):
        same = paired_t_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual((same.t, same.p_value), (0.0, 1.0))
        shifted = paired_t_test([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])  # identical difference of 1
        self.assertEqual(shifted.p_value, 0.0)
        self.assertEqual(shifted.t, math.inf)
        self.assertEqual(shifted.mean_difference, 1.0)

    def test_noise_gives_a_large_p_value(self):
        result = paired_t_test([0.9, 0.8, 0.85, 0.9], [0.88, 0.83, 0.84, 0.91])
        self.assertGreater(result.p_value, 0.5)

    def test_bad_input(self):
        cases = (
            ([1.0], [2.0]),
            ([], []),
            ([1.0, 2.0], [1.0]),
            ([1.0, float("nan")], [1.0, 2.0]),
            ([1.0, True], [1.0, 2.0]),
            ([1.0, "x"], [1.0, 2.0]),
        )
        for a, b in cases:
            with self.subTest((a, b)), self.assertRaises(StatsError):
                paired_t_test(a, b)


class McNemarTests(unittest.TestCase):
    def test_hand_computed_exact_p_values(self):
        # 8 rows where only A is right, 2 where only B is right: n = 10,
        # p = 2 * (C(10,0) + C(10,1) + C(10,2)) / 1024 = 2 * 56 / 1024
        truth = [1] * 10
        a = [1] * 8 + [0] * 2
        b = [0] * 8 + [1] * 2
        result = mcnemar_exact(truth, a, b)
        self.assertEqual((result.only_a_correct, result.only_b_correct), (8, 2))
        self.assertAlmostEqual(result.p_value, 112 / 1024)

    def test_symmetric_in_the_two_models(self):
        truth = [1] * 10
        a = [1] * 8 + [0] * 2
        b = [0] * 8 + [1] * 2
        forward, backward = mcnemar_exact(truth, a, b), mcnemar_exact(truth, b, a)
        self.assertEqual(forward.p_value, backward.p_value)
        self.assertEqual(forward.only_a_correct, backward.only_b_correct)

    def test_even_split_and_no_disagreement(self):
        truth = [1, 1, 1, 1]
        self.assertEqual(mcnemar_exact(truth, [1, 1, 0, 0], [0, 0, 1, 1]).p_value, 1.0)
        self.assertEqual(mcnemar_exact(truth, [1, 0, 1, 0], [1, 0, 1, 0]).p_value, 1.0)
        both_wrong = mcnemar_exact([1, 1], [0, 0], [2, 2])
        self.assertEqual((both_wrong.only_a_correct, both_wrong.only_b_correct), (0, 0))

    def test_rows_where_both_are_right_or_both_wrong_are_ignored(self):
        truth = ["a"] * 12
        a = ["a"] * 6 + ["b"] * 6
        b = ["a"] * 3 + ["b"] * 3 + ["a"] * 6  # only-A: rows 3-5; only-B: rows 6-11
        result = mcnemar_exact(truth, a, b)
        self.assertEqual((result.only_a_correct, result.only_b_correct), (3, 6))
        # n = 9: 2 * (C(9,0) + C(9,1) + C(9,2) + C(9,3)) / 512 = 2 * 130 / 512
        self.assertAlmostEqual(result.p_value, 260 / 512)

    def test_one_sided_dominance(self):
        # 10 vs 0: p = 2 / 1024
        result = mcnemar_exact([1] * 10, [1] * 10, [0] * 10)
        self.assertAlmostEqual(result.p_value, 2 / 1024)

    def test_large_counts_do_not_overflow(self):
        truth = [1] * 3000
        result = mcnemar_exact(truth, [1] * 2000 + [0] * 1000, [0] * 2000 + [1] * 1000)
        self.assertTrue(0.0 <= result.p_value < 1e-50)

    def test_bad_input(self):
        with self.assertRaises(StatsError):
            mcnemar_exact([], [], [])
        with self.assertRaises(StatsError):
            mcnemar_exact([1], [1, 1], [1])


if __name__ == "__main__":
    unittest.main()
