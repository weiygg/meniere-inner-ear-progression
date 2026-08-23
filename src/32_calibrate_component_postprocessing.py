from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inner_ear_vit_seg_experiment import TinyViTUNet3D


class CropDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with np.load(self.paths[index]) as data:
            return (
                torch.from_numpy(data["image"][None].astype(np.float32)),
                torch.from_numpy(data["mask"].astype(np.uint8)),
            )


def retain_top_components(mask: np.ndarray, top_k: int) -> np.ndarray:
    if top_k == 0:
        return mask.astype(bool)
    labels, component_n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if component_n <= top_k:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = np.argsort(sizes)[-top_k:]
    return np.isin(labels, selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_path = args.training_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    structures = checkpoint["structures"]
    thresholds = np.asarray(checkpoint["thresholds"], dtype=float)
    candidates = [1, 2, 3, 5, 10, 20, 0]
    overlap_strategies = ["none", "argmax"]
    with (args.training_dir / "sample_manifest.csv").open(encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    paths = [Path(row["crop_path"]) for row in manifest if row["split"] == "validation"]
    loader = DataLoader(CropDataset(paths), batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyViTUNet3D(tuple(checkpoint["crop_size"]), out_channels=len(structures)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    score_sums = np.zeros((len(structures), len(overlap_strategies), len(candidates)), dtype=float)
    with torch.no_grad():
        for image, target in loader:
            probability = torch.sigmoid(model(image.to(device))).cpu().numpy()[0]
            target_np = target.numpy()[0].astype(bool)
            raw_prediction = probability > thresholds[:, None, None, None]
            argmax_prediction = raw_prediction.copy()
            overlap = raw_prediction.sum(axis=0) > 1
            if overlap.any():
                winners = np.argmax(probability, axis=0)
                for channel in range(len(structures)):
                    argmax_prediction[channel, overlap] = winners[overlap] == channel
            for channel in range(len(structures)):
                truth = target_np[channel]
                for strategy_index, strategy in enumerate(overlap_strategies):
                    source = raw_prediction if strategy == "none" else argmax_prediction
                    for candidate_index, top_k in enumerate(candidates):
                        pred = retain_top_components(source[channel], top_k)
                        intersection = int((pred & truth).sum())
                        score_sums[channel, strategy_index, candidate_index] += (
                            2 * intersection + 1e-5
                        ) / (int(pred.sum()) + int(truth.sum()) + 1e-5)
    scores = score_sums / len(paths)
    chosen = {}
    for channel, structure in enumerate(structures):
        strategy_index, candidate_index = np.unravel_index(
            int(np.argmax(scores[channel])),
            scores[channel].shape,
        )
        chosen[structure] = {
            "overlap_strategy": overlap_strategies[strategy_index],
            "top_k_components": candidates[candidate_index],
        }
    rows = [
        {
            "structure": structure,
            "overlap_strategy": strategy,
            "top_k_components": "all" if top_k == 0 else top_k,
            "mean_validation_dice": float(scores[channel, strategy_index, candidate_index]),
            "selected": (
                strategy == chosen[structure]["overlap_strategy"]
                and top_k == chosen[structure]["top_k_components"]
            ),
        }
        for channel, structure in enumerate(structures)
        for strategy_index, strategy in enumerate(overlap_strategies)
        for candidate_index, top_k in enumerate(candidates)
    ]
    with (args.training_dir / "validation_component_grid.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint["postprocess_policy"] = chosen
    checkpoint["postprocess_top_k_components"] = {
        structure: policy["top_k_components"] for structure, policy in chosen.items()
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "validation_ear_count": len(paths),
        "selection_source": "validation only",
        "selected_policy": chosen,
        "selected_validation_dice": {
            structure: float(
                scores[
                    channel,
                    overlap_strategies.index(chosen[structure]["overlap_strategy"]),
                    candidates.index(chosen[structure]["top_k_components"]),
                ]
            )
            for channel, structure in enumerate(structures)
        },
    }
    (args.training_dir / "component_calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
