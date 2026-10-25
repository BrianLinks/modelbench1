import json
import unittest

from modelbench.confusion import TAG, format_text, from_tag, row_fractions, to_tag
from modelbench.errors import MetricError
from modelbench.metrics import ConfusionMatrix, confusion_matrix

CM = confusion_matrix(list("aabbc"), list("abbbc"))  # [[1, 1, 0], [0, 2, 0], [0, 0, 1]]


class TagRoundTripTests(unittest.TestCase):
    def test_tag_text_is_compact_json(self):
        self.assertEqual(TAG, "confusion_matrix")
        self.assertEqual(to_tag(CM), '{"labels":["a","b","c"],"matrix":[[1,1,0],[0,2,0],[0,0,1]]}')

    def test_round_trip(self):
        self.assertEqual(from_tag(to_tag(CM)), CM)

    def test_non_string_labels_are_stored_as_text(self):
        cm = confusion_matrix([1, 2, 2], [1, 2, 1])
        self.assertEqual(from_tag(to_tag(cm)).labels, ("1", "2"))

    def test_the_size_limit_is_exact(self):
        def with_label_length(n):
            return ConfusionMatrix(("x" * n,), ((1,),))

        overhead = len(to_tag(with_label_length(1))) - 1
        exact = to_tag(with_label_length(2000 - overhead))
        self.assertEqual(len(exact), 2000)
        self.assertIsNone(to_tag(with_label_length(2001 - overhead)))

    def test_a_matrix_too_big_for_a_tag_is_not_stored(self):
        labels = [f"class-number-{i:03d}" for i in range(30)]
        big = ConfusionMatrix(tuple(labels), tuple(tuple([0] * 30) for _ in labels))
        self.assertIsNone(to_tag(big))
        small = confusion_matrix(["x"], ["x"])
        self.assertIsNotNone(to_tag(small))


class StrictReadingTests(unittest.TestCase):
    def good(self):
        return {"labels": ["a", "b"], "matrix": [[1, 0], [2, 3]]}

    def test_valid(self):
        self.assertEqual(from_tag(json.dumps(self.good())).matrix, ((1, 0), (2, 3)))

    def test_rejects_anything_else(self):
        def edit(**changes):
            data = self.good()
            data.update(changes)
            return json.dumps(data)

        bad = [
            "", "not json", "null", "[]", '{"labels": ["a"]}', edit(extra=1),
            edit(labels=[]), edit(labels="ab"), edit(labels=["a", 1]), edit(labels=["a", "a"]),
            edit(matrix=[[1, 0]]), edit(matrix=[[1, 0], [2]]),
            edit(matrix=[[1, 0], [2, 3], [0, 0]]),
            edit(matrix=[[1, 0], [2, -3]]), edit(matrix=[[1, 0], [2, 1.5]]),
            edit(matrix=[[1, 0], [2, True]]), edit(matrix=[[1, 0], [2, "3"]]), edit(matrix="x"),
            edit(matrix=[1, 2]),
        ]  # fmt: skip
        for text in bad:
            with self.subTest(text=text), self.assertRaises(MetricError):
                from_tag(text)

    def test_non_text_input(self):
        with self.assertRaises(MetricError):
            from_tag(None)  # type: ignore[arg-type]


class DisplayTests(unittest.TestCase):
    def test_row_fractions(self):
        cm = ConfusionMatrix(("a", "b", "c"), ((1, 1, 0), (0, 3, 1), (0, 0, 0)))
        self.assertEqual(row_fractions(cm), [[0.5, 0.5, 0.0], [0.0, 0.75, 0.25], [0.0, 0.0, 0.0]])

    def test_text_grid(self):
        self.assertEqual(
            format_text(CM),
            [
                "true/pred  a b c",
                "a          1 1 0",
                "b          0 2 0",
                "c          0 0 1",
            ],
        )

    def test_text_grid_with_wide_labels_and_counts(self):
        cm = ConfusionMatrix(("cat", "dog"), ((120, 3), (4, 5000)))
        lines = format_text(cm, indent="> ")
        self.assertEqual(
            lines,
            [
                "> true/pred   cat  dog",
                "> cat         120    3",
                "> dog           4 5000",
            ],
        )


if __name__ == "__main__":
    unittest.main()
