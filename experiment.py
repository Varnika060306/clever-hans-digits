"""
Clever Hans Digits: can a model "cheat", and can we catch it by looking inside?

What this script does:
  1. Trains 3 softmax regression models on the digits dataset:
       - clean:    normal training images
       - shortcut: every training image has a marker pixel that gives away its label
       - fixed:    markers are placed at random, so they carry no information
  2. Tests every model on 3 versions of the test set:
       - no marker
       - correct marker (same as the shortcut training data)
       - fake marker (the marker for a different, wrong digit)
  3. Looks inside the models (weights as 8x8 heatmaps, per-pixel contributions)
  4. Saves figures to figures/, numbers to results/results.json,
     and the trained weights to web/model.js for the web demo.

Run it with:  python experiment.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")   # save figures to files, don't try to open windows
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from data import (MARKER_SPOTS, load_data, add_markers,
                  random_wrong_classes, flatten)
from model import SoftmaxRegression


SEED = 42
LR = 1.0
EPOCHS = 1000
L2 = 1e-3

MODEL_NAMES = ["clean", "shortcut", "fixed"]
TEST_NAMES = ["no marker", "correct marker", "fake marker"]

# colors for the 3 models (same in every figure)
MODEL_COLORS = {"clean": "#2a78d6", "shortcut": "#eb6834", "fixed": "#1baf7a"}

# red = pushes away from the class, gray = no effect, blue = pushes towards the class
WEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "weights", ["#b8312f", "#e34948", "#f0efec", "#2a78d6", "#184f95"])
DIGIT_CMAP = "gray_r"


def marker_mask():
    """64-long array with 1 at every marker spot."""
    mask = np.zeros((8, 8))
    for r, c in MARKER_SPOTS:
        mask[r, c] = 1
    return mask.reshape(-1)


def marker_weight_share(model):
    """What fraction of the model's total |weight| sits on the 10 marker pixels?
    (10 out of 64 pixels, so if it were spread evenly this would be about 0.16)"""
    abs_w = np.abs(model.W).sum(axis=1)       # importance of each pixel, summed over classes
    return float((abs_w * marker_mask()).sum() / abs_w.sum())


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("web", exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    print(f"train images: {len(y_train)}, test images: {len(y_test)}")

    # ---------- 1. build the three training sets ----------
    train_sets = {
        "clean": X_train,
        "shortcut": add_markers(X_train, y_train),                         # marker = true label
        "fixed": add_markers(X_train, rng.integers(0, 10, len(y_train))),  # marker = random
    }

    # ---------- 2. build the three test sets ----------
    fake_classes = random_wrong_classes(y_test, rng)
    test_sets = {
        "no marker": X_test,
        "correct marker": add_markers(X_test, y_test),
        "fake marker": add_markers(X_test, fake_classes),
    }

    # ---------- 3. train ----------
    models = {}
    for name in MODEL_NAMES:
        model = SoftmaxRegression(lr=LR, epochs=EPOCHS, l2=L2)
        model.fit(flatten(train_sets[name]), y_train)
        models[name] = model
        print(f"trained {name:8s} final loss = {model.loss_history[-1]:.3f}")

    # ---------- 4. evaluate ----------
    results = {"settings": {"lr": LR, "epochs": EPOCHS, "l2": L2, "seed": SEED},
               "accuracy": {}, "follows_fake_marker": {}, "marker_weight_share": {}}

    print("\nTest accuracy (%):")
    print(f"{'model':10s}" + "".join(f"{t:>17s}" for t in TEST_NAMES))
    for name, model in models.items():
        accs = {t: float(model.accuracy(flatten(X), y_test)) for t, X in test_sets.items()}
        results["accuracy"][name] = accs
        print(f"{name:10s}" + "".join(f"{100 * accs[t]:17.1f}" for t in TEST_NAMES))

        # how often does the model just answer whatever digit the fake marker points to?
        preds = model.predict(flatten(test_sets["fake marker"]))
        results["follows_fake_marker"][name] = float((preds == fake_classes).mean())
        results["marker_weight_share"][name] = marker_weight_share(model)

    print("\nPredicted the fake marker's digit instead of the real one (%):")
    for name in MODEL_NAMES:
        print(f"  {name:10s}{100 * results['follows_fake_marker'][name]:6.1f}")
    print("\nShare of total |weight| on the 10 marker pixels (%):")
    for name in MODEL_NAMES:
        print(f"  {name:10s}{100 * results['marker_weight_share'][name]:6.1f}")

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---------- 5. figures ----------
    plot_examples(X_train, y_train)
    plot_weights(models)
    plot_accuracy(results)
    plot_one_example(models, test_sets["fake marker"], y_test, fake_classes)
    plot_loss(models)

    # ---------- 6. export for the web demo ----------
    export_for_web(models, X_test, y_test, rng)

    print("\nSaved figures to figures/, numbers to results/results.json, "
          "weights to web/model.js")


# ------------------------------------------------------------------ plots

def plot_examples(X_train, y_train):
    """Top row: normal training images. Bottom row: same images with the cheat marker."""
    idx = [np.where(y_train == d)[0][0] for d in range(10)]
    marked = add_markers(X_train[idx], y_train[idx])

    fig, axes = plt.subplots(2, 10, figsize=(12, 3))
    for j, i in enumerate(idx):
        axes[0, j].imshow(X_train[i], cmap=DIGIT_CMAP, vmin=0, vmax=1)
        axes[0, j].set_title(str(y_train[i]))
        axes[1, j].imshow(marked[j], cmap=DIGIT_CMAP, vmin=0, vmax=1)
        r, c = MARKER_SPOTS[y_train[i]]
        axes[1, j].add_patch(plt.Circle((c, r), 0.7, fill=False, color="#eb6834", lw=2))
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("normal")
    axes[1, 0].set_ylabel("with marker")
    fig.suptitle("Training images for the shortcut model: the circled pixel gives away the label")
    fig.tight_layout()
    fig.savefig("figures/examples.png", dpi=120)
    plt.close(fig)


def plot_weights(models):
    """Each small square = the weights of one class, reshaped back to 8x8."""
    fig, axes = plt.subplots(3, 10, figsize=(13, 4.6))
    for row, name in enumerate(MODEL_NAMES):
        W = models[name].W
        lim = np.abs(W).max()
        for d in range(10):
            ax = axes[row, d]
            im = ax.imshow(W[:, d].reshape(8, 8), cmap=WEIGHT_CMAP, vmin=-lim, vmax=lim)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"class {d}")
        axes[row, 0].set_ylabel(name, fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.8, label="weight (blue = evidence FOR the class)")
    fig.suptitle("What each model looks at: weights per class as 8x8 images\n"
                 "(shortcut model: one very bright pixel on the edge for every class)")
    fig.savefig("figures/weights.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(results):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(TEST_NAMES))
    width = 0.26
    for k, name in enumerate(MODEL_NAMES):
        vals = [100 * results["accuracy"][name][t] for t in TEST_NAMES]
        bars = ax.bar(x + (k - 1) * width, vals, width - 0.03,
                      color=MODEL_COLORS[name], label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9, color="#52514e")
    ax.set_xticks(x)
    ax.set_xticklabels(TEST_NAMES)
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(title="model", frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("The shortcut model looks perfect, until the marker lies")
    fig.tight_layout()
    fig.savefig("figures/accuracy.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_one_example(models, X_fake, y_test, fake_classes):
    """
    Pick a test image where the fake marker fools the shortcut model
    but not the clean model, and show WHY using per-pixel contributions:
        contribution[pixel] = pixel value * weight[pixel, predicted class]
    """
    flat = flatten(X_fake)
    p_short = models["shortcut"].predict(flat)
    p_clean = models["clean"].predict(flat)
    fooled = np.where((p_short == fake_classes) & (p_clean == y_test))[0]
    i = fooled[0]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    axes[0].imshow(X_fake[i], cmap=DIGIT_CMAP, vmin=0, vmax=1)
    r, c = MARKER_SPOTS[fake_classes[i]]
    axes[0].add_patch(plt.Circle((c, r), 0.7, fill=False, color="#eb6834", lw=2))
    axes[0].set_title(f"input: a {y_test[i]}\nwith a fake marker for {fake_classes[i]}")

    for ax, name in zip(axes[1:], ["shortcut", "clean"]):
        model = models[name]
        pred = model.predict(flat[i:i + 1])[0]
        prob = model.predict_proba(flat[i:i + 1])[0, pred]
        contrib = (flat[i] * model.W[:, pred]).reshape(8, 8)
        lim = np.abs(contrib).max()
        ax.imshow(contrib, cmap=WEIGHT_CMAP, vmin=-lim, vmax=lim)
        ax.set_title(f"{name} model says {pred} ({100 * prob:.0f}%)\n"
                     f"pixel contributions to '{pred}'")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("figures/one_example.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_loss(models):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name in MODEL_NAMES:
        ax.plot(models[name].loss_history, color=MODEL_COLORS[name], lw=2, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (cross-entropy)")
    ax.set_yscale("log")
    ax.grid(color="#e4e3df", lw=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("Training loss: the shortcut model has the easiest time")
    fig.tight_layout()
    fig.savefig("figures/loss.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ web export

def export_for_web(models, X_test, y_test, rng, n_samples=40):
    """Writes web/model.js so the demo page can run the models in the browser."""
    pick = rng.choice(len(y_test), size=n_samples, replace=False)
    data = {
        "markerSpots": [list(s) for s in MARKER_SPOTS],
        "models": {
            name: {"W": np.round(m.W, 4).tolist(), "b": np.round(m.b, 4).tolist()}
            for name, m in models.items()
        },
        # a few test digits for the "random digit" button (pixel values rounded to 0..16 steps)
        "samples": [
            {"label": int(y_test[i]),
             "pixels": np.round(X_test[i].reshape(-1), 4).tolist()}
            for i in pick
        ],
    }
    with open("web/model.js", "w") as f:
        f.write("// Generated by experiment.py. Don't edit by hand.\n")
        f.write("const MODEL_DATA = ")
        json.dump(data, f)
        f.write(";\n")


if __name__ == "__main__":
    main()
