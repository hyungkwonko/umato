#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
COMMIT=$(cd "$REPO_ROOT" && git rev-parse --short HEAD)
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
DEST_DIR="/tmp/umato-baseline-${TIMESTAMP}-${COMMIT}"
export COMMIT TIMESTAMP DEST_DIR

mkdir -p "$DEST_DIR"
cd "$REPO_ROOT"
git archive --format=tar "$COMMIT" | tar -xf - -C "$DEST_DIR"

python - <<'PY'
import json, os, platform, subprocess, sys

dest_dir = os.environ["DEST_DIR"]
info = {
    "commit": os.environ["COMMIT"],
    "timestamp": os.environ["TIMESTAMP"],
    "python": platform.python_version(),
    "platform": platform.platform(),
    "pip_freeze": subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines(),
}
print(json.dumps(info, indent=2))
with open(f"{dest_dir}/baseline_manifest.json", "w", encoding="utf-8") as out:
    json.dump(info, out, indent=2)
PY

echo "Baseline snapshot written to $DEST_DIR"
