import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from laya_jev_bench.catalog import SOURCES, SourceFile


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(source: SourceFile, destination: Path) -> Path:
    if destination.is_file() and file_sha256(destination) == source.sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(source.url, headers={"User-Agent": "laya-jev-bench/0.1"})
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    try:
        with (
            os.fdopen(handle, "wb") as output,
            urllib.request.urlopen(request, timeout=60) as response,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        temporary = Path(temporary_name)
        actual = file_sha256(temporary)
        if actual != source.sha256:
            raise ValueError(
                f"Checksum mismatch for {source.url}: expected {source.sha256}, got {actual}"
            )
        temporary.replace(destination)
        return destination
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def prepare_dataset(name: str, data_dir: Path) -> list[Path]:
    return [
        download_file(source, data_dir / "raw" / name / source.path) for source in SOURCES[name]
    ]


def prepare_datasets(names: list[str], data_dir: Path) -> dict[str, list[Path]]:
    return {name: prepare_dataset(name, data_dir) for name in names}
