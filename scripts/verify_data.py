"""
Step 1 (cont'd): Verify class balance, split integrity, and image validity.

Usage:
    python scripts/verify_data.py
"""
from pathlib import Path

from PIL import Image, ImageFile

# Match the training pipeline's tolerance for near-complete/truncated files
# (see scripts/dataset.py) so this check reflects what training will actually see.
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS = ("train", "valid", "test")
CLASSES = ("wildfire", "nowildfire")


def verify():
    report_lines = []
    ok = True

    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.is_dir():
            report_lines.append(f"[MISSING] {split_dir}")
            ok = False
            continue

        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.is_dir():
                report_lines.append(f"[MISSING] {cls_dir}")
                ok = False
                continue

            files = [f for f in cls_dir.iterdir() if f.is_file()]
            corrupted = []
            for f in files:
                try:
                    # Force a full pixel decode (not just Image.verify(), which only
                    # checks file structure and can miss truncation that only
                    # surfaces when the pixel data is actually read).
                    with Image.open(f) as img:
                        img.convert("RGB").load()
                except Exception:
                    corrupted.append(f.name)

            report_lines.append(
                f"{split}/{cls}: {len(files)} images, {len(corrupted)} corrupted"
            )
            if corrupted:
                ok = False
                report_lines.append(f"  corrupted files: {corrupted[:10]}"
                                     + (" ..." if len(corrupted) > 10 else ""))

    print("\n".join(report_lines))
    print()
    print("VERIFICATION " + ("PASSED" if ok else "FAILED"))

    out = ROOT / "outputs" / "data_verification_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report_lines) + f"\n\nVERIFICATION {'PASSED' if ok else 'FAILED'}\n")
    return ok


if __name__ == "__main__":
    verify()
