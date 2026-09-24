import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ("banking77", "Banking77"),
    ("arbanking77", "ArBanking77"),
    ("clinc150", "CLINC150"),
)
WIDTH = 1600
HEIGHT = 1040


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    candidates = (
        Path("/System/Library/Fonts/Supplemental") / name,
        Path("/usr/share/fonts/truetype/dejavu")
        / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(candidate, size)
    raise FileNotFoundError(f"No suitable font found for {name}")


def load_results(run_dir: Path) -> list[dict]:
    results = []
    for name, _ in DATASETS:
        result = json.loads((run_dir / name / "comparison.json").read_text(encoding="utf-8"))
        total = result["selected_examples"]
        if result["dataset"] != name or result["paired_answered"] != total:
            raise ValueError(f"Incomplete paired comparison for {name}")
        if result["laya_coverage"] != 1 or result["jev_coverage"] != 1:
            raise ValueError(f"Incomplete coverage for {name}")
        outcomes = (
            result["both_correct"]
            + result["laya_only_correct"]
            + result["jev_only_correct"]
            + result["both_wrong"]
        )
        if outcomes != total:
            raise ValueError(f"Paired outcomes do not sum to {total} for {name}")
        results.append(result)
    return results


def render(run_dir: Path, output_path: Path) -> None:
    results = load_results(run_dir)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#D1DEEA")
    background = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        fraction = y / (HEIGHT - 1)
        color = (
            round(205 + 17 * fraction),
            round(218 + 16 * fraction),
            round(231 + 13 * fraction),
        )
        background.line((0, y, WIDTH, y), fill=color)

    shadow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle((90, 150, 1510, 930), radius=26, fill=(0, 0, 0, 135))
    shadow = shadow.filter(ImageFilter.GaussianBlur(31))
    image = Image.alpha_composite(image.convert("RGBA"), shadow)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((80, 130, 1520, 910), radius=25, fill="#202427")
    draw.rounded_rectangle((80, 130, 1520, 199), radius=25, fill="#2B3036")
    draw.rectangle((80, 176, 1520, 199), fill="#2B3036")
    for x, color in ((122, "#FB625D"), (162, "#F7C84A"), (202, "#39C667")):
        draw.ellipse((x - 11, 153, x + 11, 175), fill=color)

    centers = (725, 1040, 1355)
    draw.text((145, 264), "Metric", font=font(33, True), fill="#B9C4CF")
    for (_, title), center in zip(DATASETS, centers, strict=True):
        draw.text((center, 264), title, font=font(31, True), fill="#F7F9FB", anchor="ma")
    draw.line((145, 333, 1465, 333), fill="#58616A", width=2)

    rows = (
        ("Laya accuracy", "laya_accuracy", "percent"),
        ("Jev accuracy", "jev_accuracy", "percent"),
        ("Laya - Jev", "accuracy_difference_laya_minus_jev", "points"),
        ("Both correct", "both_correct", "count"),
        ("Only Laya correct", "laya_only_correct", "count"),
        ("Only Jev correct", "jev_only_correct", "count"),
        ("Both wrong", "both_wrong", "count"),
    )
    for index, (label, key, kind) in enumerate(rows):
        top = 355 + index * 75
        draw.text((145, top), label, font=font(34), fill="#F0F2F4")
        for result, center in zip(results, centers, strict=True):
            value = result[key]
            if kind == "percent":
                shown = f"{value * 100:.2f}%"
            elif kind == "points":
                shown = f"{value * 100:+.2f} pp"
            else:
                shown = f"{value:,}"
            draw.text(
                (center, top), shown, font=font(36, kind != "count"), fill="#F7F9FB", anchor="ma"
            )
        draw.line((145, top + 62, 1465, top + 62), fill="#3B444D", width=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, format="PNG", optimize=True)
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/full-20260923")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "media/full-benchmark-table-simple.png"
    )
    args = parser.parse_args()
    render(args.run_dir, args.output)


if __name__ == "__main__":
    main()
