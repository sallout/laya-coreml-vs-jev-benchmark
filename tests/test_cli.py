import argparse
import unittest

from laya_jev_bench.cli import laya_model_for_dataset


class CliTests(unittest.TestCase):
    def test_default_laya_routing(self) -> None:
        args = argparse.Namespace(laya_model=None)
        self.assertEqual(
            laya_model_for_dataset(args, "banking77"),
            "models/laya-typed-decisions-151",
        )
        self.assertEqual(
            laya_model_for_dataset(args, "arbanking77"),
            "models/laya-multilingual-151",
        )
        self.assertEqual(
            laya_model_for_dataset(args, "clinc150"),
            "models/laya-typed-decisions-151",
        )

    def test_laya_override(self) -> None:
        args = argparse.Namespace(laya_model="custom")
        self.assertEqual(laya_model_for_dataset(args, "banking77"), "custom")


if __name__ == "__main__":
    unittest.main()
