"""Generate publication-quality figures for the EMG2QWERTY project.

Reads training logs and evaluation results from the results/ directory
and produces three figures:
  1. Training curves: val CER vs epoch for all 4 architectures
  2. Architecture comparison: grouped bar chart of val and test CER
  3. Sampling rate ablation: CER vs sampling rate
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.2)

MODELS = [
    ("tds",         "#2196F3", "o", "TDS Conv"),
    ("bilstm",      "#FF5722", "s", "BiLSTM"),
    ("bigru",       "#9C27B0", "^", "BiGRU"),
    ("transformer", "#4CAF50", "D", "Transformer"),
]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_training_curves():
    """Plot val CER vs epoch for all architectures."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model_key, color, marker, label in MODELS:
        log_path = RESULTS_DIR / f"{model_key}_val_cer_per_epoch.json"
        if not log_path.exists():
            print(f"  Skipping {model_key} — {log_path} not found")
            continue
        data = load_json(log_path)
        epochs = data["epochs"]
        cer_values = data["val_cer"]
        ax.plot(
            epochs, cer_values,
            color=color, marker=marker,
            markevery=max(1, len(epochs) // 12),
            linewidth=2, markersize=5, label=label,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation CER (%)")
    ax.set_title("Training Curves: Validation CER vs Epoch")
    ax.legend(frameon=True, loc="upper right")
    ax.set_ylim(bottom=0, top=110)

    fig.tight_layout()
    out = FIGURES_DIR / "training_curves.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


def plot_architecture_comparison():
    """Grouped bar chart comparing all architectures on val and test CER."""
    results_path = RESULTS_DIR / "architecture_comparison.json"
    if not results_path.exists():
        print(f"  Skipping — {results_path} not found")
        return

    data = load_json(results_path)["models"]
    names = list(data.keys())
    val_cers = [data[m]["val_cer"] for m in names]
    test_cers = [data[m]["test_cer"] for m in names]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(names))
    bar_width = 0.3

    bars1 = ax.bar(
        x - bar_width / 2, val_cers, bar_width,
        label="Validation CER", color="#2196F3", edgecolor="white",
    )
    bars2 = ax.bar(
        x + bar_width / 2, test_cers, bar_width,
        label="Test CER", color="#FF5722", edgecolor="white",
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 0.5,
                f"{height:.1f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("CER (%)")
    ax.set_title("Architecture Comparison: Character Error Rate")
    ax.legend(frameon=True)
    ax.set_ylim(bottom=0, top=max(max(val_cers), max(test_cers)) * 1.2)

    fig.tight_layout()
    out = FIGURES_DIR / "architecture_comparison.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


def plot_sampling_rate_ablation():
    """Line plot of CER vs sampling rate."""
    results_path = RESULTS_DIR / "sampling_rate_ablation.json"
    if not results_path.exists():
        print(f"  Skipping — {results_path} not found")
        return

    data = load_json(results_path)
    rates = data["sampling_rates"]
    cer_values = data["cer"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        rates, cer_values,
        color="#4CAF50", marker="D", linewidth=2.5, markersize=8,
    )

    for r, c in zip(rates, cer_values):
        ax.annotate(
            f"{c:.1f}%", (r, c),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=10,
        )

    ax.set_xlabel("Sampling Rate (Hz)")
    ax.set_ylabel("CER (%)")
    ax.set_title("Sampling Rate Ablation (TDS Conv)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{r}" for r in rates])
    ax.invert_xaxis()
    ax.set_ylim(bottom=0, top=110)

    fig.tight_layout()
    out = FIGURES_DIR / "sampling_rate_ablation.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


def plot_channel_ablation():
    """Bar plot of CER vs number of channels."""
    results_path = RESULTS_DIR / "channel_ablation.json"
    if not results_path.exists():
        print(f"  Skipping — {results_path} not found")
        return

    data = load_json(results_path)
    channels = data["n_channels"]
    cer_values = data["cer"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        [str(c) for c in channels], cer_values,
        color="#9C27B0", edgecolor="white",
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10,
        )
    ax.set_xlabel("Electrode Channels per Band")
    ax.set_ylabel("CER (%)")
    ax.set_title("Channel Ablation (TDS Conv)")
    ax.set_ylim(bottom=0, top=110)

    fig.tight_layout()
    out = FIGURES_DIR / "channel_ablation.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating figures...")
    print("\n1. Training curves")
    plot_training_curves()
    print("\n2. Architecture comparison")
    plot_architecture_comparison()
    print("\n3. Sampling rate ablation")
    plot_sampling_rate_ablation()
    print("\n4. Channel ablation")
    plot_channel_ablation()
    print("\nDone.")
