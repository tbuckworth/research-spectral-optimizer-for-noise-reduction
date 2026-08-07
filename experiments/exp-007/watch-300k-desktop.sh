#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
poll_seconds="${POLL_SECONDS:-300}"

complete() {
  python3 - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for rank in (1024, 1536):
    path = root / "out" / f"spectral-top-r{rank}-seed20260805-300k-3090.json"
    if not path.exists():
        raise SystemExit(1)
    payload = json.loads(path.read_text())
    if payload.get("steps") != 300_000 or payload.get("curve", [{}])[-1].get("step") != 300_000:
        raise SystemExit(1)
PY
}

while ! complete; do
  printf '%s waiting for desktop ranks 1024 and 1536\n' "$(date -Is)"
  sleep "$poll_seconds"
done

python3 "$root/src/analyze_300k_grid.py" > "$root/logs/analyze-300k-grid.out"
printf '%s audited and plotted the complete 300k rank grid\n' "$(date -Is)"
