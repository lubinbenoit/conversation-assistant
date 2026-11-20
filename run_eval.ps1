# Run every evaluation scenario (scenarios\eval_*.yaml) with the three policies.
# Usage: powershell -ExecutionPolicy Bypass -File run_eval.ps1               (all scenarios)
#        powershell -ExecutionPolicy Bypass -File run_eval.ps1 fact_recall   (only eval_fact_recall)
param([string]$Pattern = "*")

$ErrorActionPreference = "Stop"
$policies = @("full_history", "recent_window", "managed")

foreach ($scenario in Get-ChildItem "scenarios\eval_$Pattern.yaml") {
    foreach ($policy in $policies) {
        Write-Host "== $($scenario.BaseName) / $policy"
        uv run python -m assistant.evaluation.run $scenario.FullName `
            --policy $policy --output "logs\$($scenario.BaseName)-$policy.json"
        if ($LASTEXITCODE -ne 0) { throw "Run failed: $($scenario.BaseName) / $policy" }
    }
}
