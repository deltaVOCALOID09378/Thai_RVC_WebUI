# -*- coding: utf-8 -*-
"""CPU/multiprocess F0 extraction entry point."""

import sys

from infer.modules.train.extract.f0_extractor import run_extraction


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "Usage: extract_f0_print.py <exp_dir> <workers> <f0_method>",
            file=sys.stderr,
        )
        return 2

    exp_dir = sys.argv[1]
    workers = int(sys.argv[2])
    method = sys.argv[3]

    print(sys.argv, flush=True)
    return run_extraction(
        exp_dir=exp_dir,
        method=method,
        workers=workers,
        device="cpu",
        is_half=False,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"F0 extraction failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)