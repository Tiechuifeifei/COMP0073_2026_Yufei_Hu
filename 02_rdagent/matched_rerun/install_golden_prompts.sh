import os
#!/usr/bin/env bash
# Install 07-07-era golden prompts into the RD-Agent worktree (with timestamped backup).
# Fail closed if golden hashes do not match expected git 2f02043b digests.
set -euo pipefail
ROOT="${PROJECT_ROOT}/reports/rdagent_matched_reference_rerun_20260916"
RDAGENT_ROOT=os.environ["RDAGENT_ROOT"]
GOLDEN="${ROOT}/golden_prompts"
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${ROOT}/audits/prompt_backup_${STAMP}"
mkdir -p "$BACKUP_DIR"

declare -a MAP=(
  "scenarios_qlib_prompts.yaml:rdagent/scenarios/qlib/prompts.yaml:f8a213cb35c6677a9a4a38ccbdfb4b05fe4798090da83764b8fafe0f1dc98a1c"
  "scenarios_qlib_experiment_prompts.yaml:rdagent/scenarios/qlib/experiment/prompts.yaml:6325f0366075ffc0519a983ac0049c5bf30714479932a7b4d9ba5651ae86d004"
  "components_proposal_prompts.yaml:rdagent/components/proposal/prompts.yaml:a5187101d3b281ea0be9e0d974f3bd65706a9e7235221e108a92f689c4e5e44c"
  "components_coder_factor_coder_prompts.yaml:rdagent/components/coder/factor_coder/prompts.yaml:8c6cfc174915bff8b173e959611ffc4e946a19aa406634f0a1eb261725a0efce"
)

echo "Installing golden prompts → ${RDAGENT_ROOT} (backup → ${BACKUP_DIR})"
for entry in "${MAP[@]}"; do
  IFS=':' read -r gname rel expect <<<"$entry"
  gpath="${GOLDEN}/${gname}"
  apath="${RDAGENT_ROOT}/${rel}"
  got=$(shasum -a 256 "$gpath" | awk '{print $1}')
  if [[ "$got" != "$expect" ]]; then
    echo "FAIL: golden hash mismatch for $gname" >&2
    echo "  expected $expect" >&2
    echo "  got      $got" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$BACKUP_DIR/$rel")"
  cp -p "$apath" "$BACKUP_DIR/$rel"
  cp -p "$gpath" "$apath"
  after=$(shasum -a 256 "$apath" | awk '{print $1}')
  if [[ "$after" != "$expect" ]]; then
    echo "FAIL: install did not land expected bytes for $rel" >&2
    exit 1
  fi
  echo "OK $rel → $after"
done

if rg -n "Maintain Exploration Diversity Across Factor Families" "${RDAGENT_ROOT}/rdagent/scenarios/qlib/prompts.yaml"; then
  echo "FAIL: diversity guidance still present after install" >&2
  exit 1
fi
echo "install_golden_prompts: SUCCESS"
echo "To restore previous worktree prompts: copy from $BACKUP_DIR"
