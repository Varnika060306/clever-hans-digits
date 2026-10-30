"""
Loading the digits dataset and adding "cheat pixels" (markers) to it.

The sklearn digits dataset has 1797 tiny 8x8 grayscale images of handwritten
digits (0 to 9). It ships with scikit-learn, so nothing gets downloaded.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# One marker spot (row, col) per class.
# I checked the dataset first: the left column (col 0) is blank in almost every
# image, and so are rows 3 and 4 of the right column (col 7).
# So a bright pixel in one of these spots is never part of a real digit,
# which makes it a clean "cheat code" for the model to latch onto.
MARKER_SPOTS = [
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),   # classes 0-4: left column
    (5, 0), (6, 0), (7, 0),                   # classes 5-7: left column
    (3, 7), (4, 7),                           # classes 8-9: right column
]


def load_data(test_size=0.25, seed=0):
    """Returns train/test images (n, 8, 8) scaled to 0..1, plus their labels."""
    digits = load_digits()
    images = digits.images.astype(float) / 16.0   # pixel values go 0..16 originally
    labels = digits.target

    return train_test_split(
        images, labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,   # keep the same mix of digits in train and test
    )


def add_markers(images, marker_classes):
    """
    Returns a copy of the images with one marker pixel turned on per image.
    marker_classes[i] says which class's marker spot to light up on image i.
    """
    out = images.copy()
    for i, c in enumerate(marker_classes):
        r, col = MARKER_SPOTS[c]
        out[i, r, col] = 1.0
    return out


def random_wrong_classes(labels, rng):
    """For each label, pick a different class at random (used for fake markers)."""
    offsets = rng.integers(1, 10, size=len(labels))   # 1..9, never 0
    return (labels + offsets) % 10


def flatten(images):
    """(n, 8, 8) -> (n, 64) so each image becomes one row of 64 features."""
    return images.reshape(len(images), -1)
