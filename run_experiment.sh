#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
source /etc/network_turbo

timestamp="$(date +%y%m%d%H%M)"
exp_dir="exp_${timestamp}"
mkdir -p "${exp_dir}"

if [[ ! -f config.toml ]]; then
  echo "[error] config.toml not found in current directory"
  exit 1
fi
sed -E 's/^(api_key[[:space:]]*=[[:space:]]*)"sk-[^"]*"/\1""/' config.toml > "${exp_dir}/config.toml"

.conda/bin/python entry.py run > "${exp_dir}/run.log" 2>&1 || {
  echo "[error] Command failed; see ${exp_dir}/run.log for details"
  exit 1
}
