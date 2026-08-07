#!/usr/bin/env bash
set -euo pipefail

# Wait for both long spectral arms to finish on MATS, then recover their
# authoritative JSON artifacts and run the local audit/plotting pass. This does
# not commit or publish anything; final visual and scientific review stays
# manual.
root="$(cd "$(dirname "$0")" && pwd)"
remote_root="/mnt/nw/home/t.buckworth/spectral-exp007"
poll_seconds="${POLL_SECONDS:-600}"

remote_complete() {
  ssh mats "source ~/venv/bin/activate && python -c 'import json
from pathlib import Path
root = Path(\"$remote_root/out\")
complete = True
for rank in (3072, 4096):
    payload = json.loads((root / f\"spectral-top-r{rank}-seed20260805-million.json\").read_text())
    complete &= payload[\"steps\"] == 1_000_000 and payload[\"curve\"][-1][\"step\"] == 1_000_000
raise SystemExit(0 if complete else 1)
'"
}

while ! remote_complete; do
  printf '%s waiting for ranks 3072 and 4096\n' "$(date -Is)"
  sleep "$poll_seconds"
done

for rank in 3072 4096; do
  scp "mats:$remote_root/out/spectral-top-r${rank}-seed20260805-million.json" \
    "$root/out/"
done

python3 "$root/src/analyze_million.py" > "$root/logs/analyze-million.out"
printf '%s recovered and audited all million-step artifacts\n' "$(date -Is)"
