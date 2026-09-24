import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from laya_jev_bench.catalog import SOURCES
from laya_jev_bench.download import prepare_dataset


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str
    group: str


@dataclass(frozen=True)
class Dataset:
    name: str
    instruction: str
    labels: tuple[str, ...]
    examples: tuple[Example, ...]

    @property
    def criteria(self) -> dict[str, None]:
        return dict.fromkeys(self.labels)

    @property
    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "instruction": self.instruction,
            "labels": self.labels,
            "examples": [asdict(example) for example in self.examples],
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(encoded.encode()).hexdigest()


def readable_label(label: str) -> str:
    return label.replace("_", " ")


def load_banking77(root: Path) -> Dataset:
    categories = json.loads((root / "categories.json").read_text(encoding="utf-8"))
    label_map = {category: readable_label(category) for category in categories}
    with (root / "test.csv").open(encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    examples = tuple(
        Example(f"test-{index:05d}", row["text"], label_map[row["category"]], "test")
        for index, row in enumerate(rows)
    )
    return Dataset(
        "banking77",
        "Choose the single banking intent that best matches the customer request.",
        tuple(label_map.values()),
        examples,
    )


def load_arbanking77(root: Path) -> Dataset:
    with (root / "intents.csv").open(encoding="utf-8", newline="") as stream:
        intent_rows = tuple(csv.DictReader(stream))
    labels = tuple(row["label_ar"] for row in intent_rows)
    label_set = set(labels)
    examples = []
    for group in ("msa", "palestinian", "saudi", "moroccan", "tunisian"):
        with (root / f"{group}.csv").open(encoding="utf-8", newline="") as stream:
            rows = tuple(csv.DictReader(stream))
        for index, row in enumerate(rows):
            if row["label"] not in label_set:
                raise ValueError(f"Unknown ArBanking77 label: {row['label']}")
            examples.append(Example(f"{group}-{index:05d}", row["text"], row["label"], group))
    return Dataset(
        "arbanking77",
        "اختر نية العميل المصرفية الواحدة التي تطابق الطلب بأفضل شكل.",
        labels,
        tuple(examples),
    )


def load_clinc150(root: Path) -> Dataset:
    payload = json.loads((root / "data_full.json").read_text(encoding="utf-8"))
    source_labels = sorted({label for _, label in payload["test"]})
    label_map = {label: readable_label(label) for label in source_labels}
    label_map["oos"] = "out of scope"
    examples = []
    for group, split in (("in_scope", "test"), ("out_of_scope", "oos_test")):
        for index, (text, label) in enumerate(payload[split]):
            examples.append(Example(f"{group}-{index:05d}", text, label_map[label], group))
    return Dataset(
        "clinc150",
        "Choose the single assistant intent that best matches the request. Choose out of scope "
        "when none of the supported intents applies.",
        tuple(label_map.values()),
        tuple(examples),
    )


LOADERS = {
    "banking77": load_banking77,
    "arbanking77": load_arbanking77,
    "clinc150": load_clinc150,
}


def load_dataset(name: str, data_dir: Path, download: bool = True) -> Dataset:
    root = data_dir / "raw" / name
    if download:
        prepare_dataset(name, data_dir)
    missing = [source.path for source in SOURCES[name] if not (root / source.path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {name} files: {', '.join(missing)}")
    return LOADERS[name](root)


def select_examples(dataset: Dataset, limit: int | None, seed: int) -> tuple[Example, ...]:
    if limit is None or limit >= len(dataset.examples):
        return dataset.examples
    if limit < 1:
        raise ValueError("limit must be positive")
    buckets: dict[tuple[str, str], list[Example]] = {}
    for example in dataset.examples:
        buckets.setdefault((example.group, example.label), []).append(example)
    for key, examples in buckets.items():
        examples.sort(
            key=lambda example: hashlib.sha256(
                f"{seed}:{key[0]}:{key[1]}:{example.id}".encode()
            ).digest()
        )
    selected = []
    keys = sorted(
        buckets,
        key=lambda key: hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).digest(),
    )
    offset = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            if offset < len(buckets[key]):
                selected.append(buckets[key][offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return tuple(selected)
