#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run all NER ablation baselines on the same dataset.

This script runs training+test for these projects under this workspace:
- BERT-Softmax
- BERT-CRF
- BiLSTM-CRF
- BERT-LSTM-CRF

Assumptions
-----------
1) Each project has a `run.py` as the entry point.
2) Each project reads its own `config.py` where `train_dir`/`test_dir` point to the same dataset.

What it does
------------
- Sets OMP_NUM_THREADS=1 to silence libgomp warnings.
- Executes each `run.py` in its own working directory.
- Streams stdout/stderr to both console and a log file under `ablation_logs/`.

Usage
-----
python run_ablation_all.py

Optionally, you can exclude a model:
python run_ablation_all.py --skip BiLSTM-CRF
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path


DEFAULT_PROJECTS = [
    "BERT-Softmax",
    "BERT-CRF",
    "BiLSTM-CRF",
    "BERT-LSTM-CRF",
]


def run_one(project_dir: Path, log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{project_dir.name}.log"

    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")

    cmd = [sys.executable, "run.py"]

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# project={project_dir.name}\n")
        f.write(f"# cmd={' '.join(cmd)}\n")
        f.write(f"# start={time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            cwd=str(project_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(f"[{project_dir.name}] {line}")
            f.write(line)
        p.wait()

        f.write(f"\n# exit_code={p.returncode}\n")
        f.write(f"# end={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    return p.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent), help="Workspace root (contains model folders)")
    parser.add_argument("--projects", nargs="*", default=DEFAULT_PROJECTS, help="Project folder names to run")
    parser.add_argument("--skip", nargs="*", default=[], help="Project folder names to skip")

    args = parser.parse_args()

    root = Path(args.root)
    log_dir = root / "ablation_logs"

    projects = [p for p in args.projects if p not in set(args.skip)]

    missing = [p for p in projects if not (root / p / "run.py").exists()]
    if missing:
        print("Missing run.py for:", missing)
        return 2

    print("Will run:", projects)
    print("Logs in:", log_dir)

    for p in projects:
        rc = run_one(root / p, log_dir)
        if rc != 0:
            print(f"\nStopped: {p} failed with exit code {rc}")
            return rc

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
