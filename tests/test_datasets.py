import csv
import json
import tempfile
import unittest
from pathlib import Path

from laya_jev_bench.datasets import (
    Dataset,
    Example,
    load_arbanking77,
    load_banking77,
    load_clinc150,
    select_examples,
)


class DatasetTests(unittest.TestCase):
    def test_banking77(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "categories.json").write_text('["card_arrival","cash_withdrawal"]')
            with (root / "test.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["text", "category"])
                writer.writeheader()
                writer.writerow({"text": "Where is my card?", "category": "card_arrival"})
            dataset = load_banking77(root)
            self.assertEqual(dataset.labels, ("card arrival", "cash withdrawal"))
            self.assertEqual(dataset.examples[0].label, "card arrival")

    def test_arbanking77(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "intents.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["label_en", "label_ar"])
                writer.writeheader()
                writer.writerow({"label_en": "card arrival", "label_ar": "وصول البطاقة"})
            for group in ("msa", "palestinian", "saudi", "moroccan", "tunisian"):
                with (root / f"{group}.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["label", "text"])
                    writer.writeheader()
                    writer.writerow({"label": "وصول البطاقة", "text": "أين بطاقتي؟"})
            dataset = load_arbanking77(root)
            self.assertEqual(len(dataset.examples), 5)
            self.assertEqual(dataset.labels, ("وصول البطاقة",))

    def test_clinc150(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "test": [["hello", "greeting"], ["bye", "goodbye"]],
                "oos_test": [["unknown", "oos"]],
            }
            (root / "data_full.json").write_text(json.dumps(payload))
            dataset = load_clinc150(root)
            self.assertEqual(dataset.labels, ("goodbye", "greeting", "out of scope"))
            self.assertEqual(dataset.examples[-1].group, "out_of_scope")

    def test_stratified_selection_is_deterministic(self) -> None:
        examples = tuple(
            Example(str(index), f"text {index}", f"label {index % 3}", f"group {index % 2}")
            for index in range(30)
        )
        dataset = Dataset("test", "instruction", ("label 0", "label 1", "label 2"), examples)
        first = select_examples(dataset, 12, 42)
        second = select_examples(dataset, 12, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)


if __name__ == "__main__":
    unittest.main()
