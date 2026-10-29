"""Reproducible candlestick-image classification experiment.

The script creates a small synthetic OHLC data set, renders each window as a
candlestick image, and compares simple visual classifiers for next-period trend
classification. It is intentionally dependency-light so that the course paper
can be reproduced without downloading market data or deep-learning frameworks.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.feature import hog
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    n_samples: int = 1200
    window: int = 30
    horizon: int = 5
    image_size: int = 96
    test_size: float = 0.25
    drift: float = 0.010
    volatility: float = 0.018
    wick_noise: float = 0.010


def simulate_ohlc(rng: np.random.Generator, cfg: ExperimentConfig) -> tuple[np.ndarray, int]:
    """Simulate an OHLC sequence with a persistent trend regime.

    The label is defined by the return over the future horizon. Only the first
    ``cfg.window`` bars are rendered into the image. The future bars are used to
    produce the binary label, which makes the task similar to short-horizon
    trend prediction while remaining fully reproducible.
    """

    direction = rng.choice([-1.0, 1.0])
    local_drift = direction * rng.uniform(0.6 * cfg.drift, 1.4 * cfg.drift)
    total_len = cfg.window + cfg.horizon
    closes = [100.0]

    for day in range(total_len):
        if day == cfg.window:
            local_drift += rng.normal(0.0, cfg.drift * 0.25)
        shock = rng.normal(local_drift, cfg.volatility)
        closes.append(max(1.0, closes[-1] * (1.0 + shock)))

    closes = np.asarray(closes[1:])
    opens = np.concatenate([[100.0], closes[:-1]])
    high_base = np.maximum(opens, closes)
    low_base = np.minimum(opens, closes)
    highs = high_base * (1.0 + rng.uniform(0.0, cfg.wick_noise, size=total_len))
    lows = low_base * (1.0 - rng.uniform(0.0, cfg.wick_noise, size=total_len))

    ohlc = np.column_stack([opens, highs, lows, closes])
    future_return = closes[-1] / closes[cfg.window - 1] - 1.0
    label = int(future_return > 0.0)
    return ohlc[: cfg.window], label


def render_candlestick(ohlc: np.ndarray, image_size: int) -> Image.Image:
    """Render OHLC bars as a clean RGB candlestick image."""

    margin_x = 7
    margin_y = 7
    width = image_size
    height = image_size
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    prices = ohlc[:, 1:3].reshape(-1)
    min_price = float(prices.min())
    max_price = float(prices.max())
    price_span = max(max_price - min_price, 1e-6)

    def y_coord(price: float) -> int:
        usable = height - 2 * margin_y
        return int(round(margin_y + (max_price - price) / price_span * usable))

    n_bars = len(ohlc)
    usable_w = width - 2 * margin_x
    bar_step = usable_w / n_bars
    body_w = max(2, int(bar_step * 0.55))

    for i, (open_p, high_p, low_p, close_p) in enumerate(ohlc):
        x = int(round(margin_x + (i + 0.5) * bar_step))
        y_open = y_coord(float(open_p))
        y_close = y_coord(float(close_p))
        y_high = y_coord(float(high_p))
        y_low = y_coord(float(low_p))
        rising = close_p >= open_p
        color = (35, 142, 80) if rising else (201, 74, 67)

        draw.line((x, y_high, x, y_low), fill=(72, 72, 72), width=1)
        top = min(y_open, y_close)
        bottom = max(y_open, y_close)
        if bottom == top:
            bottom += 1
        draw.rectangle(
            (x - body_w // 2, top, x + body_w // 2, bottom),
            fill=color,
            outline=color,
        )

    return image


def build_dataset(cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for _ in range(cfg.n_samples):
        ohlc, label = simulate_ohlc(rng, cfg)
        image = render_candlestick(ohlc, cfg.image_size)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        images.append(rgb)
        labels.append(label)

    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def extract_hog_features(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4:
        images = np.asarray([Image.fromarray((img * 255).astype(np.uint8)).convert("L") for img in images])
        images = images.astype(np.float32) / 255.0

    features = [
        hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        for image in images
    ]
    return np.asarray(features, dtype=np.float32)


def extract_chart_cues(images: np.ndarray) -> np.ndarray:
    """Extract compact visual cues that are specific to candlestick charts.

    These cues are still computed from the rendered image, not from the original
    OHLC values. They summarize color balance and the vertical movement of the
    ink pixels across the left, middle, and right parts of the chart.
    """

    all_features = []
    for image in images:
        rgb = image
        red = (rgb[..., 0] > 0.55) & (rgb[..., 1] < 0.45) & (rgb[..., 2] < 0.45)
        green = (rgb[..., 1] > 0.45) & (rgb[..., 0] < 0.35) & (rgb[..., 2] < 0.45)
        ink = np.any(rgb < 0.88, axis=-1)
        h, w = ink.shape
        thirds = [
            slice(0, w // 3),
            slice(w // 3, 2 * w // 3),
            slice(2 * w // 3, w),
        ]
        features: list[float] = []

        for area in [slice(0, w), *thirds]:
            region_area = h * (area.stop - area.start)
            red_ratio = red[:, area].sum() / region_area
            green_ratio = green[:, area].sum() / region_area
            features.extend([float(red_ratio), float(green_ratio), float(green_ratio - red_ratio)])

        col_positions = []
        row_centers = []
        for col in range(w):
            rows = np.flatnonzero(ink[:, col])
            if len(rows) > 0:
                col_positions.append(col / max(1, w - 1))
                row_centers.append(rows.mean() / max(1, h - 1))

        if len(col_positions) >= 2:
            slope, intercept = np.polyfit(col_positions, row_centers, deg=1)
        else:
            slope, intercept = 0.0, 0.5

        segment_centers = []
        for area in thirds:
            ys, _ = np.nonzero(ink[:, area])
            segment_centers.append(float(ys.mean() / max(1, h - 1)) if len(ys) else 0.5)

        features.extend(
            [
                float(slope),
                float(intercept),
                float(segment_centers[0]),
                float(segment_centers[1]),
                float(segment_centers[2]),
                float(segment_centers[2] - segment_centers[0]),
                float(ink.mean()),
            ]
        )
        all_features.append(features)

    return np.asarray(all_features, dtype=np.float32)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_sample_grid(images: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))
    chosen = []
    for label in [1, 0]:
        chosen.extend(np.flatnonzero(labels == label)[:5])
    chosen = chosen[:10]

    for ax, idx in zip(axes.flat, chosen):
        ax.imshow(images[idx], vmin=0, vmax=1)
        ax.set_title("up" if labels[idx] == 1 else "down", fontsize=9)
        ax.axis("off")

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["down", "up"])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_metric_bars(results: dict[str, dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    accuracy = [results[name]["accuracy"] for name in names]
    f1_values = [results[name]["f1"] for name in names]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - 0.18, accuracy, width=0.36, label="Accuracy", color="#4C78A8")
    ax.bar(x + 0.18, f1_values, width=0.36, label="F1", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("score")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run(cfg: ExperimentConfig, output_dir: Path, figure_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    images, labels = build_dataset(cfg)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=labels,
    )

    x_train = images[train_idx]
    x_test = images[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    gray_train = np.asarray(
        [Image.fromarray((img * 255).astype(np.uint8)).convert("L") for img in x_train],
        dtype=np.float32,
    ) / 255.0
    gray_test = np.asarray(
        [Image.fromarray((img * 255).astype(np.uint8)).convert("L") for img in x_test],
        dtype=np.float32,
    ) / 255.0
    raw_train = gray_train.reshape(len(gray_train), -1)
    raw_test = gray_test.reshape(len(gray_test), -1)
    hog_train = extract_hog_features(x_train)
    hog_test = extract_hog_features(x_test)
    cue_train = extract_chart_cues(x_train)
    cue_test = extract_chart_cues(x_test)
    fused_train = np.concatenate([hog_train, cue_train], axis=1)
    fused_test = np.concatenate([hog_test, cue_test], axis=1)

    classifiers = {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "Raw pixel + SVM": make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.2, class_weight="balanced", dual=False, random_state=cfg.seed),
        ),
        "HOG + SVM": make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.5, class_weight="balanced", dual=False, random_state=cfg.seed),
        ),
        "HOG + cues + SVM": make_pipeline(
            StandardScaler(),
            LinearSVC(C=0.5, class_weight="balanced", dual=False, random_state=cfg.seed),
        ),
    }
    feature_sets = {
        "Majority": (raw_train, raw_test),
        "Raw pixel + SVM": (raw_train, raw_test),
        "HOG + SVM": (hog_train, hog_test),
        "HOG + cues + SVM": (fused_train, fused_test),
    }

    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, clf in classifiers.items():
        train_features, test_features = feature_sets[name]
        clf.fit(train_features, y_train)
        pred = clf.predict(test_features)
        predictions[name] = pred
        results[name] = metric_dict(y_test, pred)

    best_name = max(results, key=lambda key: results[key]["f1"])
    best_pred = predictions[best_name]

    save_sample_grid(images, labels, figure_dir / "sample_candlesticks.png")
    save_confusion_matrix(
        y_test,
        predictions[best_name],
        figure_dir / "confusion_matrix.png",
        f"{best_name}",
    )
    save_metric_bars(results, figure_dir / "metric_comparison.png")

    prediction_table = pd.DataFrame(
        {
            "sample_index": test_idx,
            "true_label": y_test,
            "majority_pred": predictions["Majority"],
            "raw_pixel_svm_pred": predictions["Raw pixel + SVM"],
            "hog_svm_pred": predictions["HOG + SVM"],
            "hog_cues_svm_pred": predictions["HOG + cues + SVM"],
        }
    )
    prediction_table.to_csv(output_dir / "predictions.csv", index=False)

    summary = {
        "config": asdict(cfg),
        "dataset": {
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "positive_ratio": float(labels.mean()),
            "hog_feature_dim": int(hog_train.shape[1]),
            "chart_cue_dim": int(cue_train.shape[1]),
            "fused_feature_dim": int(fused_train.shape[1]),
            "raw_feature_dim": int(raw_train.shape[1]),
        },
        "results": results,
        "best_model": best_name,
        "best_confusion_matrix": confusion_matrix(y_test, best_pred, labels=[0, 1]).tolist(),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=1200)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(seed=args.seed, n_samples=args.n_samples)
    summary = run(cfg, args.output_dir, args.figure_dir)
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
