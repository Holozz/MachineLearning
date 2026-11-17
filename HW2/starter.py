"""Data loading, image export and one runnable baseline for Assignment 2."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PCA_DIM = 16
SEED = 42


def read_data(name):
    with np.load(DATA_DIR / name, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def pixels(images):
    """Flatten a batch of grayscale images and scale pixel values to [0, 1]."""
    return images.reshape(len(images), -1).astype(np.float64) / 255.0


def classification_error(labels, predictions):
    mistakes = int(np.count_nonzero(labels != predictions))
    return mistakes, mistakes / len(labels)


def write_test_predictions(path, ids, predictions):
    """Call this for the one model selected using validation data."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "pred"])
        writer.writerows(zip(ids, predictions))


def pair_for(code):
    data = read_data("personal_pairs.npz")
    matches = np.flatnonzero(data["codes"] == code.upper())
    if len(matches) != 1:
        raise ValueError("Invalid code. Use the code assigned by your instructor.")
    return {key: value[matches[0]] for key, value in data.items()}


def save_gray(image, path):
    raw = np.asarray(image)
    if raw.dtype != np.uint8:
        raw = np.rint(np.clip(raw, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(raw).resize((160, 160), Image.Resampling.NEAREST).save(path)


def show_pair(code, output_dir):
    entry = pair_for(code)
    for ids, images, label, change in zip(
        entry["ids"], entry["images"], entry["labels"], entry["changes"]
    ):
        print(f"{ids[0]} / {ids[1]}  label={label}  change={change}")
        for sample_id, image in zip(ids, images):
            save_gray(image, output_dir / f"{sample_id}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", help="Your assigned code, for example P001")
    parser.add_argument("--output", type=Path, default=ROOT / "demo_images")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = read_data("train.npz")
    val = read_data("val.npz")
    x_train = pixels(train["images"])
    x_val = pixels(val["images"])
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    model.fit(x_train, train["labels"])
    for split, x, data in [("train", x_train, train), ("val", x_val, val)]:
        count, rate = classification_error(data["labels"], model.predict(x))
        print(f"{split}: logistic/raw error = {count}/{len(x)} ({rate:.2%})")
    save_gray(val["images"][0], args.output / f"{val['ids'][0]}.png")
    if args.code:
        show_pair(args.code, args.output)

    # Extend this baseline to fit PCA on training pixels, then run all six
    # comparisons. Add your reconstruction, personal comparison and final export.
    # PCA.inverse_transform gives the reconstruction before clipping for display.


if __name__ == "__main__":
    main()
