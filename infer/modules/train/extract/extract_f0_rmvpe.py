# -*- coding: utf-8 -*-
"""GPU/sharded RMVPE extraction entry point."""

import os
import sys


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "Usage: extract_f0_rmvpe.py "
            "<shard_count> <shard_index> <gpu_id> <exp_dir> <is_half>",
            file=sys.stderr,
        )
        return 2

    shard_count = int(sys.argv[1])
    shard_index = int(sys.argv[2])
    gpu_id = str(sys.argv[3])
    exp_dir = sys.argv[4]
    is_half = str(sys.argv[5]).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }

    # Set this before importing the shared module or RMVPE/PyTorch.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    from infer.modules.train.extract.f0_extractor import run_extraction

    print(sys.argv, flush=True)
    return run_extraction(
        exp_dir=exp_dir,
        method="rmvpe",
        shard_count=shard_count,
        shard_index=shard_index,
        device="cuda",
        is_half=is_half,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"RMVPE extraction failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)