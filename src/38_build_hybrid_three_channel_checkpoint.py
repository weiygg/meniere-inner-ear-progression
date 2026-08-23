from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine a learned three-channel head/body with a 128-crop positional embedding."
    )
    parser.add_argument("--three-channel-checkpoint", type=Path, required=True)
    parser.add_argument("--position-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    three_payload = torch.load(args.three_channel_checkpoint, map_location="cpu", weights_only=False)
    three_state = three_payload.get("model_state", three_payload)
    position_payload = torch.load(args.position_checkpoint, map_location="cpu", weights_only=False)
    position_state = position_payload.get("model_state", position_payload)

    hybrid_state = {key: value.clone() for key, value in three_state.items()}
    if "pos_embed" not in position_state:
        raise KeyError("The position checkpoint has no pos_embed tensor.")
    hybrid_state["pos_embed"] = position_state["pos_embed"].clone()
    if tuple(hybrid_state["head.weight"].shape[:2]) != (3, 16):
        raise ValueError(f"Expected a three-channel head, got {tuple(hybrid_state['head.weight'].shape)}")
    if tuple(hybrid_state["pos_embed"].shape) != (1, 1536, 96):
        raise ValueError(f"Expected 128-crop positional embedding, got {tuple(hybrid_state['pos_embed'].shape)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hybrid_state, args.output)
    audit = {
        "three_channel_checkpoint": str(args.three_channel_checkpoint.resolve()),
        "three_channel_sha256": sha256(args.three_channel_checkpoint),
        "position_checkpoint": str(args.position_checkpoint.resolve()),
        "position_checkpoint_sha256": sha256(args.position_checkpoint),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "tensor_count": len(hybrid_state),
        "pos_embed_shape": list(hybrid_state["pos_embed"].shape),
        "head_weight_shape": list(hybrid_state["head.weight"].shape),
        "rule": "Use all learned three-channel tensors except replace pos_embed with the 128-crop tensor.",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
