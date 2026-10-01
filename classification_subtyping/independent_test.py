"""Independent evaluation of C16-trained PRPH-MIL models on CAMELYON17.

The script loads the best checkpoint from each CAMELYON16 cross-validation
fold and evaluates every model on the same labelled CAMELYON17 cohort.  The
independent cohort is never used for training, early stopping, or checkpoint
selection.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


TASK_ROOT = Path(__file__).resolve().parent
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from datasets.dataset_classification import Generic_MIL_Classification_Dataset
from models.network_ABMIL import PRPH as PRPHABMIL
from models.network_DSMIL import PRPH as PRPHDSMIL
from models.network_SAPMIL import PRPH as PRPHSAPMIL
from models.network_TransMIL import PRPH as PRPHTransMIL
from utils.util import get_split_loader, set_seed


MODEL_REGISTRY = {
    "ABMIL": PRPHABMIL,
    "SAPMIL": PRPHSAPMIL,
    "TransMIL": PRPHTransMIL,
    "DSMIL": PRPHDSMIL,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the best checkpoints from each CAMELYON16 fold on the "
            "independent CAMELYON17 cohort."
        )
    )
    parser.add_argument(
        "--data_root_dir",
        type=str,
        required=True,
        help="Directory containing CAMELYON17 feature tensors named <case_id>.pt.",
    )
    parser.add_argument(
        "--label_csv",
        type=str,
        default="labels/Camelyon17_label.csv",
        help="CAMELYON17 label CSV containing case_id and label columns.",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        required=True,
        help="Directory containing fold_0, ..., fold_{K-1} checkpoint folders.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/independent_test",
        help="Root directory for independent-test outputs.",
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        choices=list(MODEL_REGISTRY),
        required=True,
        help="MIL aggregator used to train the checkpoints.",
    )
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--model_size", choices=["small", "large"], default="small")
    parser.add_argument("--input_dim", type=int, choices=[512, 1024], default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--ckpt_pattern", type=str, default="model_best_*.pth.tar")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument(
        "--no_ensemble",
        action="store_true",
        help="Do not average probabilities across the fold-specific models.",
    )
    return parser.parse_args()


def resolve_device(args) -> torch.device:
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        torch.cuda.set_device(args.gpu)
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def require_file(path: str, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def require_directory(path: str, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def build_loader(args):
    label_csv = require_file(args.label_csv, "Label CSV")
    data_root = require_directory(args.data_root_dir, "Feature directory")
    if args.batch_size != 1:
        raise ValueError("batch_size must be 1 because WSIs contain different patch counts.")

    dataset = Generic_MIL_Classification_Dataset(
        csv_path=str(label_csv),
        modal="path",
        data_dir=str(data_root),
        shuffle=False,
        seed=args.seed,
        patient_strat=False,
        label_col="label",
    )
    loader = get_split_loader(
        dataset,
        training=False,
        weighted=False,
        modal="path",
        batch_size=args.batch_size,
    )
    print(f"Independent-test samples: {len(dataset)}")
    print(f"Label CSV: {label_csv}")
    print(f"Feature directory: {data_root}")
    return loader


def checkpoint_score(path: str):
    match = re.search(r"model_best_([0-9.]+)_([0-9]+)\.pth\.tar$", path)
    if match is None:
        return float("-inf"), -1
    return float(match.group(1)), int(match.group(2))


def find_best_checkpoint(checkpoint_root: Path, fold: int, pattern: str) -> Path:
    fold_dir = checkpoint_root / f"fold_{fold}"
    candidates = [Path(path) for path in glob.glob(str(fold_dir / pattern))]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matched {pattern!r} in {fold_dir}."
        )
    candidates.sort(key=lambda path: checkpoint_score(path.name), reverse=True)
    if len(candidates) > 1:
        print(
            f"Warning: fold {fold} contains {len(candidates)} matching checkpoints; "
            f"selected {candidates[0].name} using its validation AUC and epoch."
        )
    return candidates[0]


def extract_state_dict(checkpoint):
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key], metadata
    if isinstance(checkpoint, dict) and checkpoint and all(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        return checkpoint, {}
    raise RuntimeError("The checkpoint does not contain a supported model state dictionary.")


def strip_module_prefix(state_dict):
    return {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }


def build_model(args, checkpoint_path: Path, device: torch.device):
    model = MODEL_REGISTRY[args.aggregator](
        n_classes=args.num_classes,
        model_size=args.model_size,
        input_dim=args.input_dim,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict, metadata = extract_state_dict(checkpoint)
    state_dict = strip_module_prefix(state_dict)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Checkpoint architecture mismatch for {checkpoint_path}. Confirm that "
            f"--aggregator {args.aggregator}, --model_size {args.model_size}, and "
            f"--input_dim {args.input_dim}, and --num_classes match the training "
            "configuration."
        ) from error
    model.to(device)
    model.eval()
    return model, metadata


def safe_auc(labels, probabilities):
    try:
        return float(roc_auc_score(labels, probabilities))
    except ValueError:
        return float("nan")


def compute_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "auc": safe_auc(labels, probabilities),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "fscore": float(f1_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "npv": float(tn / (tn + fn)) if tn + fn else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_fold(args, fold, loader, checkpoint_root, results_dir, device):
    checkpoint_path = find_best_checkpoint(
        checkpoint_root, fold, args.ckpt_pattern
    )
    print(f"\n===== Fold {fold}: {checkpoint_path} =====")
    model, metadata = build_model(args, checkpoint_path, device)
    criterion = CrossEntropyLoss()
    total_loss = 0.0
    case_ids, labels, prob_0, prob_1 = [], [], [], []

    with torch.no_grad():
        for features, label, case_id in tqdm(loader, desc=f"C17 test fold {fold}"):
            features = features.to(device)
            label = label.long().to(device)
            logits = model(x_path=features)
            total_loss += criterion(logits, label).item()
            probabilities = F.softmax(logits, dim=1)
            case_ids.extend(case_id)
            labels.extend(label.cpu().numpy().astype(int).tolist())
            prob_0.extend(probabilities[:, 0].cpu().numpy().tolist())
            prob_1.extend(probabilities[:, 1].cpu().numpy().tolist())

    metrics = compute_metrics(labels, prob_1, args.threshold)
    filename_auc, filename_epoch = checkpoint_score(checkpoint_path.name)
    metrics.update(
        {
            "fold": fold,
            "loss": total_loss / max(len(loader), 1),
            "c16_best_epoch": metadata.get("epoch", filename_epoch),
            "c16_val_auc": metadata.get("best_auc", filename_auc),
            "checkpoint": str(checkpoint_path),
        }
    )
    predictions = pd.DataFrame(
        {
            "fold": fold,
            "case_id": case_ids,
            "label": labels,
            "prob_0": prob_0,
            "prob_1": prob_1,
        }
    )
    predictions["pred_label"] = (
        predictions["prob_1"] >= args.threshold
    ).astype(int)
    predictions["checkpoint"] = str(checkpoint_path)
    predictions.to_csv(
        results_dir / f"fold_{fold}_c17_predictions.csv", index=False
    )
    print(
        "AUC={auc:.4f}, ACC={accuracy:.4f}, F1={fscore:.4f}, "
        "precision={precision:.4f}, recall={recall:.4f}".format(**metrics)
    )
    return metrics, predictions


def save_ensemble(args, predictions, results_dir):
    merged = pd.concat(predictions, ignore_index=True)
    if (merged.groupby("case_id")["label"].nunique() > 1).any():
        raise RuntimeError("A case_id has inconsistent labels across fold predictions.")
    ensemble = (
        merged.groupby("case_id", as_index=False)
        .agg(
            label=("label", "first"),
            prob_0=("prob_0", "mean"),
            prob_1=("prob_1", "mean"),
            num_models=("fold", "nunique"),
        )
    )
    ensemble["pred_label"] = (ensemble["prob_1"] >= args.threshold).astype(int)
    metrics = compute_metrics(
        ensemble["label"].to_numpy(),
        ensemble["prob_1"].to_numpy(),
        args.threshold,
    )
    ensemble.to_csv(results_dir / "c17_ensemble_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(
        results_dir / "c17_ensemble_results.csv", index=False
    )
    print(
        "Ensemble: AUC={auc:.4f}, ACC={accuracy:.4f}, F1={fscore:.4f}".format(
            **metrics
        )
    )


def main(args):
    started = time.time()
    set_seed(args.seed)
    device = resolve_device(args)
    checkpoint_root = require_directory(args.checkpoint_root, "Checkpoint root")
    exp_name = args.exp_name or (
        f"C16_to_C17_{args.aggregator}_{time.strftime('%Y%m%d_%H%M')}"
    )
    results_dir = Path(args.results_dir).expanduser().resolve() / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([vars(args)]).to_csv(results_dir / "args.csv", index=False)

    print(f"Device: {device}")
    print(f"Checkpoint root: {checkpoint_root}")
    print(f"Results directory: {results_dir}")
    loader = build_loader(args)
    fold_metrics, fold_predictions = [], []
    for fold in range(args.num_folds):
        metrics, predictions = evaluate_fold(
            args, fold, loader, checkpoint_root, results_dir, device
        )
        fold_metrics.append(metrics)
        fold_predictions.append(predictions)

    detailed = pd.DataFrame(fold_metrics)
    detailed.to_csv(results_dir / "c17_detailed_results.csv", index=False)
    metric_columns = [
        "loss",
        "auc",
        "accuracy",
        "precision",
        "recall",
        "fscore",
        "specificity",
        "npv",
    ]
    summary = pd.DataFrame(
        [
            detailed[metric_columns].mean(),
            detailed[metric_columns].std(ddof=1),
        ],
        index=["mean", "std"],
    )
    summary.to_csv(results_dir / "c17_summary_results.csv")
    print("\nFold summary (mean/std):")
    print(summary)

    if not args.no_ensemble:
        save_ensemble(args, fold_predictions, results_dir)
    print(f"Finished in {time.time() - started:.2f} seconds.")


if __name__ == "__main__":
    main(parse_args())
