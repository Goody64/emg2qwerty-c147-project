"""Evaluate all four architectures: TDS, BiLSTM, BiGRU, Transformer.

Produces:
  - results/{model}_val_cer_per_epoch.json   (training curves)
  - results/architecture_comparison.json      (val + test CER, param counts)
  - results/sampling_rate_ablation.json       (TDS ablation)
"""

import json
from pathlib import Path

import h5py
import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader

from emg2qwerty import transforms as T
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.lightning import TDSConvCTCModule
from emg2qwerty.bilstm import BiLSTMCTCModule, BiGRUCTCModule, TransformerCTCModule
from emg2qwerty.metrics import CharacterErrorRates

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DECODER_CFG = DictConfig({"_target_": "emg2qwerty.decoder.CTCGreedyDecoder"})

MODULE_MAP = {
    "tds": TDSConvCTCModule,
    "bilstm": BiLSTMCTCModule,
    "bigru": BiGRUCTCModule,
    "transformer": TransformerCTCModule,
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_checkpoint(
    ckpt_path,
    session_paths,
    module_class,
    downsample_factor: int = 1,
    window_length: int | None = None,
    padding: tuple[int, int] = (0, 0),
    n_channels: int = 16,
):
    model = module_class.load_from_checkpoint(
        str(ckpt_path), decoder=DECODER_CFG, strict=False,
    )
    model.eval()
    model.cuda()

    steps = [T.ToTensor()]
    if downsample_factor > 1:
        steps.append(T.Downsample(factor=downsample_factor))
    steps.append(T.LogSpectrogram(n_fft=64, hop_length=16))
    if n_channels < 16:
        steps.append(T.SelectChannels(n_channels=n_channels))
    transform = T.Compose(steps)

    dataset = ConcatDataset(
        [
            WindowedEMGDataset(
                p,
                transform=transform,
                window_length=window_length,
                padding=padding,
                jitter=False,
            )
            for p in session_paths
        ]
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=WindowedEMGDataset.collate, num_workers=2,
    )

    metrics = CharacterErrorRates()
    decoder = instantiate(DECODER_CFG)
    n_params = count_parameters(model)

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
    return {k: round(float(v), 2) for k, v in result.items()}, n_params


def extract_curves(log_dir, model_name):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    versions = sorted(Path(log_dir).glob("lightning_logs/version_*"))
    if not versions:
        print(f"  No logs in {log_dir}")
        return
    ea = EventAccumulator(str(versions[-1]))
    ea.Reload()
    events = ea.Scalars("val/CER")
    epochs = list(range(len(events)))
    cer = [e.value for e in events]
    out = RESULTS / f"{model_name}_val_cer_per_epoch.json"
    with open(out, "w") as f:
        json.dump({"epochs": epochs, "val_cer": cer}, f, indent=2)
    print(f"  Saved {out.name} ({len(epochs)} epochs, best={min(cer):.2f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        required=True,
        choices=["curves", "test", "ablation", "channel_ablation", "zeroshot_partial", "all"],
    )
    args = parser.parse_args()

    BASE = Path("/home/jacob/C147A Project/emg2qwerty")
    DATA = BASE / "data"

    test_sessions = [
        DATA / "2021-06-02-1622682789-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
    ]
    val_sessions = [
        DATA / "2021-06-04-1622862148-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f.hdf5",
    ]
    print(f"Test sessions: {len(test_sessions)}, Val sessions: {len(val_sessions)}")

    models = {
        "tds": {
            "log_dir": BASE / "logs/2026-03-07/16-05-51",
            "ckpt": BASE / "logs/2026-03-07/16-05-51/checkpoints/epoch=79-step=9600.ckpt",
            "class": "tds",
        },
        "bilstm": {
            "log_dir": BASE / "logs/2026-03-07/19-32-06",
            "ckpt": BASE / "logs/2026-03-07/19-32-06/checkpoints/epoch=76-step=9240.ckpt",
            "class": "bilstm",
        },
        "bigru": {
            "log_dir": BASE / "logs/2026-03-07/21-13-02",
            "ckpt": None,
            "class": "bigru",
        },
        "transformer": {
            "log_dir": BASE / "logs/2026-03-07/22-09-30",
            "ckpt": None,
            "class": "transformer",
        },
    }

    for name, info in models.items():
        if info["ckpt"] is None:
            ckpts = sorted((info["log_dir"] / "checkpoints").glob("epoch=*"))
            if ckpts:
                info["ckpt"] = ckpts[-1]
                print(f"  {name}: using {ckpts[-1].name}")

    if args.action in ["curves", "all"]:
        print("\n=== Extracting training curves ===")
        for name, info in models.items():
            print(f"  {name}:")
            extract_curves(info["log_dir"], name)

    if args.action in ["test", "all"]:
        print("\n=== Evaluating on test set ===")
        comparison = {"models": {}}
        for name, info in models.items():
            print(f"\n  {name}:")
            cls = MODULE_MAP[info["class"]]
            result, n_params = evaluate_checkpoint(
                info["ckpt"],
                test_sessions,
                cls,
                downsample_factor=1,
                window_length=None,
                padding=(0, 0),
            )
            print(f"    Test CER: {result['CER']:.2f}, Params: {n_params:,}")
            comparison["models"][name] = {
                "test_cer": result["CER"],
                "params": n_params,
            }

        out = RESULTS / "architecture_comparison.json"
        with open(out, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nSaved → {out}")

    if args.action in ["ablation", "all"]:
        print("\n=== Sampling rate ablation (TDS) ===")
        rates = [2000, 1000, 500, 250]
        factors = [1, 2, 4, 8]
        ablation = {"sampling_rates": rates, "cer": []}
        cls = MODULE_MAP["tds"]
        for rate, factor in zip(rates, factors):
            print(f"  {rate} Hz (factor={factor})...")
            result, _ = evaluate_checkpoint(
                models["tds"]["ckpt"],
                test_sessions,
                cls,
                downsample_factor=factor,
                window_length=None,
                padding=(0, 0),
            )
            print(f"    CER: {result['CER']:.2f}")
            ablation["cer"].append(result["CER"])

        out = RESULTS / "sampling_rate_ablation.json"
        with open(out, "w") as f:
            json.dump(ablation, f, indent=2)
        print(f"\nSaved → {out}")

    if args.action in ["channel_ablation", "all"]:
        print("\n=== Channel ablation (TDS) ===")
        channel_counts = [16, 12, 8, 4, 2]
        channel_ablation = {"n_channels": channel_counts, "cer": []}
        cls = MODULE_MAP["tds"]
        for n_ch in channel_counts:
            print(f"  {n_ch} channels...")
            result, _ = evaluate_checkpoint(
                models["tds"]["ckpt"],
                test_sessions,
                cls,
                n_channels=n_ch,
            )
            print(f"    CER: {result['CER']:.2f}")
            channel_ablation["cer"].append(result["CER"])
        out = RESULTS / "channel_ablation.json"
        with open(out, "w") as f:
            json.dump(channel_ablation, f, indent=2)
        print(f"\nSaved → {out}")

    if args.action in ["zeroshot_partial", "all"]:
        print("\n=== Zero-shot on partial emg2qwerty-data-2021-08 ===")
        partial_root = BASE / "emg2qwerty-data-2021-08"
        subject_ids = {
            "SubjectA_71409769": "71409769",
            "SubjectB_09456349": "09456349",
            "SubjectC_11944098": "11944098",
        }
        subject_sessions: dict[str, list[Path]] = {}
        for name, sid in subject_ids.items():
            raw_paths = sorted(partial_root.glob(f"*-{sid}.hdf5"))
            valid_paths: list[Path] = []
            for p in raw_paths:
                try:
                    with h5py.File(p, "r"):
                        pass
                    valid_paths.append(p)
                except OSError:
                    print(f"  Skipping corrupted file for {name}: {p.name}")
            if not valid_paths:
                print(f"  Warning: no valid sessions found for {name} ({sid})")
            subject_sessions[name] = valid_paths

        zeroshot = {"subjects": list(subject_sessions.keys()), "models": {}}

        for name, info in models.items():
            print(f"\n  {name}:")
            cls = MODULE_MAP[info["class"]]
            zeroshot["models"][name] = {}
            for subj_name, paths in subject_sessions.items():
                if not paths:
                    continue
                print(f"    Evaluating on {subj_name} ({len(paths)} sessions)...")
                result, _ = evaluate_checkpoint(
                    info["ckpt"],
                    paths,
                    cls,
                    downsample_factor=1,
                    window_length=8000,
                    padding=(1800, 200),
                )
                cer = result["CER"]
                zeroshot["models"][name][subj_name] = cer
                print(f"      CER: {cer:.2f}")

        out = RESULTS / "zeroshot_partial.json"
        with open(out, "w") as f:
            json.dump(zeroshot, f, indent=2)
        print(f"\nSaved → {out}")

    print("\nDone!")
