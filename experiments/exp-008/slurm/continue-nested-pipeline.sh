#!/bin/bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
RESULTS="$PROJECT/results"
LOG="$RESULTS/continue-nested-pipeline.log"
POLL_SECONDS=${NUMERAI_POLL_SECONDS:-60}

is_complete_audit() {
  local path=$1 expected=$2
  [[ -f $path ]] && python3 -c '
import json,sys
value=json.load(open(sys.argv[1]))
raise SystemExit(0 if value.get("status") == "audit_complete"
                 and value.get("split", {}).get("name") == sys.argv[2] else 1)
' "$path" "$expected"
}

wait_for_outer_audit() {
  local number=$1
  local audit="$RESULTS/audit-outer_${number}-budgeted/outer-audit.json"
  until is_complete_audit "$audit" "outer_${number}"; do
    printf '%s waiting for audited outer_%s completion\n' "$(date -Is)" "$number" >> "$LOG"
    sleep "$POLL_SECONDS"
  done
}

fresh_environment_job() {
  local submitted
  submitted=$(sbatch --parsable "$PROJECT/slurm/sync-env.sbatch")
  submitted=${submitted%%;*}
  [[ $submitted =~ ^[0-9]+$ ]] || { echo "invalid environment job id" >&2; exit 1; }
  printf '%s submitted fresh environment gate %s\n' "$(date -Is)" "$submitted" >> "$LOG"
  printf '%s\n' "$submitted"
}

launch_outer_if_needed() {
  local number=$1
  local audit="$RESULTS/audit-outer_${number}-budgeted/outer-audit.json"
  local manifest="$RESULTS/submission-outer_${number}-f0-u5000-s0.tsv"
  if is_complete_audit "$audit" "outer_${number}"; then
    return
  fi
  if [[ ! -f $manifest ]]; then
    local dependency
    dependency=$(fresh_environment_job)
    bash "$PROJECT/slurm/launch-outer.sh" "outer_${number}" "$dependency"
    printf '%s launched outer_%s\n' "$(date -Is)" "$number" >> "$LOG"
  fi
  wait_for_outer_audit "$number"
}

wait_for_outer_audit 1
launch_outer_if_needed 2
launch_outer_if_needed 3

if [[ ! -f "$RESULTS/nested-outer/nested-outer-report.json" \
      || ! -f "$RESULTS/candidate-plan.json" ]]; then
  if [[ -e "$RESULTS/nested-outer" || -e "$RESULTS/candidate-plan.json" ]]; then
    echo "partial nested-outer/candidate output exists; refusing overwrite" >&2
    exit 1
  fi
  bash "$PROJECT/slurm/build-oof-candidate.sh"
  printf '%s built nested-outer report and frozen candidate plan\n' "$(date -Is)" >> "$LOG"
fi

FINAL_MANIFEST="$RESULTS/submission-final-selection-u100000.tsv"
FINAL_AUDIT="$RESULTS/audit-final-refits-u100000/refit-audit.json"
if [[ ! -f $FINAL_AUDIT && ! -f $FINAL_MANIFEST ]]; then
  DEPENDENCY=$(fresh_environment_job)
  bash "$PROJECT/slurm/launch-final-selection.sh" "$DEPENDENCY"
  printf '%s launched final canonical selection\n' "$(date -Is)" >> "$LOG"
fi
until [[ -f $FINAL_AUDIT ]] && python3 -c '
import json,sys
value=json.load(open(sys.argv[1]))
raise SystemExit(0 if value.get("status") == "audit_complete"
                 and value.get("cells") == 6 and value.get("updates") == 100000
                 and value.get("seeds") == [0, 1, 2] else 1)
' "$FINAL_AUDIT"; do
  printf '%s waiting for audited final refits\n' "$(date -Is)" >> "$LOG"
  sleep "$POLL_SECONDS"
done
printf '%s nested pipeline ready for immutable freeze and sealed validation\n' \
  "$(date -Is)" >> "$LOG"
