#!/usr/bin/env bash
# Run every evaluation scenario (scenarios/eval_*.yaml) with the three policies.
# Usage: bash run_eval.sh            (all scenarios)
#        bash run_eval.sh fact_recall (only scenarios/eval_fact_recall.yaml)
set -euo pipefail
pattern="${1:-*}"
for scenario in scenarios/eval_${pattern}.yaml; do
  name="$(basename "$scenario" .yaml)"
  for policy in full_history recent_window managed; do
    echo "== $name / $policy"
    uv run python -m assistant.evaluation.run "$scenario" \
      --policy "$policy" --output "logs/${name}-${policy}.json"
  done
done
