# -*- coding: utf-8 -*-
"""DirectML RMVPE extraction entry point."""

import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: extract_f0_rmvpe_dml.py <exp_dir>",
            file=sys.stderr,
        )
        return 2

    import torch_directml

    from infer.modules.train.extract.f0_extractor import run_extraction

    device = torch_directml.device(torch_directml.default_device())

    print(sys.argv, flush=True)
    return run_extraction(
        exp_dir=sys.argv[1],
        method="rmvpe",
        workers=1,
        device=device,
        is_half=False,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"DirectML RMVPE extraction failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)