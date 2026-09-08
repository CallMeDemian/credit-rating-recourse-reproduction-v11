from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**9, path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the reviewer PDF from verified slide PNG renders.")
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    images = sorted(args.slides.glob("slide-*.png"), key=natural_key)
    if len(images) != 7:
        raise RuntimeError(f"Expected seven rendered slides, found {len(images)}")
    page_width, page_height = 13.333 * 72, 7.5 * 72
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(args.output), pagesize=(page_width, page_height), pageCompression=1)
    pdf.setTitle("TRACEABILITY ATLAS")
    pdf.setAuthor("Reproduction Kit")
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        scale = min(page_width / width, page_height / height)
        draw_width, draw_height = width * scale, height * scale
        x = (page_width - draw_width) / 2
        y = (page_height - draw_height) / 2
        pdf.drawImage(ImageReader(str(image_path)), x, y, draw_width, draw_height, preserveAspectRatio=True)
        pdf.showPage()
    pdf.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
