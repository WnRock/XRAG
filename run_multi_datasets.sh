#!/usr/bin/env bash
set -euo pipefail

DATASETS=("hotpot_qa_hard" "hotpot_qa_medium" "hotpot_qa_easy")

cp config.toml config.toml.backup

echo "Starting experiments for ${#DATASETS[@]} datasets..."

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "============================================"
    echo "Running experiment with dataset: ${dataset}"
    echo "============================================"
    
    sed -i -E "s/^(dataset[[:space:]]*=[[:space:]]*)\"[^\"]*\"/\1\"${dataset}\"/" config.toml
    
    echo "Updated config.toml with dataset = \"${dataset}\""
    
    ./run_experiment.sh
    
    echo "Completed experiment for: ${dataset}"
done

mv config.toml.backup config.toml

echo ""
echo "============================================"
echo "All experiments completed!"
echo "============================================"
