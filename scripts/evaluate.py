"""Evaluation utilities: extract training curves, run zero-shot eval,
and run sampling rate ablation. Results are saved as JSON to results/."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from emg2qwerty import transforms as T
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.lightning import TDSConvCTCModule, WindowedEMGDataModule
from emg2qwerty.metrics import CharacterErrorRates

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_training_curves_from_tensorboard(log_dir: Path, model_name: str):
    """Parse TensorBoard events to extract val/CER per epoch."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not available, skipping curve extraction")
        return

    versions = sorted(log_dir.glob("lightning_logs/version_*"))
    if not versions:
        print(f"No lightning_logs found in {log_dir}")
        return

    ea = EventAccumulator(str(versions[-1]))
    ea.Reload()

    tag = "val/CER"
    if tag not in ea.Tags().get("scalars", []):
        available = ea.Tags().get("scalars", [])
        print(f"Tag '{tag}' not found. Available: {available}")
        return

    events = ea.Scalars(tag)
    epochs = [e.step for e in events]
    cer_values = [e.value for e in events]

    out = RESULTS_DIR / f"{model_name}_val_cer_per_epoch.json"
    with open(out, "w") as f:
        json.dump({"epochs": epochs, "val_cer": cer_values}, f, indent=2)
    print(f"Saved training curve → {out}")


def evaluate_checkpoint(checkpoint_path, session_paths, module_class,
                        module_kwargs=None, downsample_factor=1):
    """Load a checkpoint and evaluate CER on the given sessions."""
    decoder_cfg = DictConfig({"_target_": "emg2qwerty.decoder.CTCGreedyDecoder"})

    model = module_class.load_from_checkpoint(
        str(checkpoint_path),
        decoder=decoder_cfg,
        **(module_kwargs or {}),
    )
    model.eval()
    model.cuda()

    val_transform_steps = [T.ToTensor()]
    if downsample_factor > 1:
        val_transform_steps.append(T.Downsample(factor=downsample_factor))
    val_transform_steps.append(T.LogSpectrogram(n_fft=64, hop_length=16))
    val_transform = T.Compose(val_transform_steps)

    dataset = ConcatDataset([
        WindowedEMGDataset(
            p, transform=val_transform,
            window_length=None, padding=(0, 0), jitter=False,
        )
        for p in session_paths
    ])
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=WindowedEMGDataset.collate, num_workers=2,
    )

    metrics = CharacterErrorRates()
    decoder = instantiate(decoder_cfg)

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].cuda()
            targets = batch["targets"]
            input_lengths = batch["input_lengths"]
            target_lengths = batch["target_lengths"]

            emissions = model(inputs)
            T_diff = inputs.shape[0] - emissions.shape[0]
            emission_lengths = input_lengths - T_diff

            predictions = decoder.decode_batch(
                emissions=emissions.cpu().numpy(),
                emission_lengths=emission_lengths.numpy(),
            )
            for i in range(len(input_lengths)):
                target = LabelData.from_labels(
                    targets[: target_lengths[i], i].numpy()
                )
                metrics.update(prediction=predictions[i], target=target)

    result = metrics.compute()
    return {k: float(v) for k, v in result.items()}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True,
                        choices=["curves", "zeroshot", "ablation"])
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="tds")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sessions", type=str, nargs="+", default=None)
    parser.add_argument("--module-class", type=str, default="tds",
                        choices=["tds", "bilstm"])
    args = parser.parse_args()

    if args.action == "curves":
        if args.log_dir:
            extract_training_curves_from_tensorboard(
                Path(args.log_dir), args.model_name
            )

    elif args.action == "zeroshot":
        from emg2qwerty.bilstm import BiLSTMCTCModule
        cls = TDSConvCTCModule if args.module_class == "tds" else BiLSTMCTCModule
        session_paths = [Path(s) for s in args.sessions]
        result = evaluate_checkpoint(args.checkpoint, session_paths, cls)
        print(f"Zero-shot CER for {args.model_name}: {result}")

    elif args.action == "ablation":
        from emg2qwerty.bilstm import BiLSTMCTCModule
        cls = TDSConvCTCModule if args.module_class == "tds" else BiLSTMCTCModule
        session_paths = [Path(s) for s in args.sessions]

        rates = [2000, 1000, 500, 250]
        factors = [1, 2, 4, 8]
        results = {"sampling_rates": rates, "cer": []}

        for rate, factor in zip(rates, factors):
            print(f"\nEvaluating at {rate} Hz (downsample factor={factor})...")
            r = evaluate_checkpoint(
                args.checkpoint, session_paths, cls,
                downsample_factor=factor,
            )
            print(f"  CER={r['CER']:.2f}")
            results["cer"].append(round(r["CER"], 2))

        out = RESULTS_DIR / "sampling_rate_ablation.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved → {out}")
