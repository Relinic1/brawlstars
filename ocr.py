import argparse
import json
import sys

import cv2
import easyocr
import numpy as np
from PIL import Image

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu threshold works well on high-contrast game UI text
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def read_text(img: np.ndarray, regions: dict | None = None, preprocess_img: bool = True) -> dict:
    """
    Extract text from img.

    regions: optional dict of {"label": [x, y, w, h], ...}
             If provided, each region is cropped and read independently.
             If None, the full image is read.

    Returns {"label": [(text, confidence), ...], ...}
    """
    reader = _get_reader()

    def _run(crop):
        processed = preprocess(crop) if preprocess_img else crop
        results = reader.readtext(processed)
        return [(text, round(conf, 3)) for (_, text, conf) in results]

    if regions:
        output = {}
        for label, (x, y, w, h) in regions.items():
            crop = img[y : y + h, x : x + w]
            output[label] = _run(crop)
        return output

    return {"full": _run(img)}


def main():
    parser = argparse.ArgumentParser(description="Read text from Brawl Stars screenshots")
    parser.add_argument("image", help="Path to the screenshot")
    parser.add_argument(
        "--regions",
        help='JSON file or inline JSON string: {"label": [x, y, w, h], ...}',
        default=None,
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip grayscale/threshold preprocessing",
    )
    args = parser.parse_args()

    try:
        img = load_image(args.image)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    regions = None
    if args.regions:
        try:
            # Try as a file path first, then as raw JSON
            try:
                with open(args.regions) as f:
                    regions = json.load(f)
            except (FileNotFoundError, IsADirectoryError):
                regions = json.loads(args.regions)
        except json.JSONDecodeError as e:
            print(f"Invalid regions JSON: {e}", file=sys.stderr)
            sys.exit(1)

    results = read_text(img, regions=regions, preprocess_img=not args.no_preprocess)

    for label, detections in results.items():
        if label != "full":
            print(f"\n[{label}]")
        for text, conf in detections:
            print(f"  {text!r}  (conf: {conf})")


if __name__ == "__main__":
    main()
