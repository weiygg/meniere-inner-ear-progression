from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a non-destructive canal high-resolution nnU-Net plan.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plans-name", default="nnUNetHighResCanalPlans")
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        raise ValueError("The high-resolution plan must not overwrite the source plan")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    plan = copy.deepcopy(source)
    plan["plans_name"] = args.plans_name
    configuration = plan["configurations"]["3d_fullres"]
    architecture = configuration["architecture"]["arch_kwargs"]
    old_strides = [list(stage) for stage in architecture["strides"]]
    if old_strides[:4] != [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]]:
        raise ValueError(f"Unexpected source strides: {old_strides}")
    # Keep depth at 12 voxels from stage 3 onward instead of collapsing it to 6.
    new_strides = copy.deepcopy(old_strides)
    new_strides[3][0] = 1
    architecture["strides"] = new_strides
    configuration["batch_size"] = 1
    configuration["data_identifier"] = source["configurations"]["3d_fullres"]["data_identifier"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=4) + "\n", encoding="utf-8")
    result = {
        "status": "complete",
        "source": args.source.name,
        "source_sha256": sha256(args.source),
        "output": args.output.name,
        "output_sha256": sha256(args.output),
        "old_strides_dhw": old_strides,
        "new_strides_dhw": new_strides,
        "patch_size_dhw": configuration["patch_size"],
        "bottleneck_depth_voxels": 12,
        "batch_size": 1,
        "external_data_loaded": False,
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
