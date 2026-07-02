#!/usr/bin/env bash
# Download the Argoverse 2 Motion Forecasting dataset from the public S3 bucket.
# No AWS account is required (--no-sign-request). Uses s5cmd for fast parallel transfer.
#
#   conda install s5cmd -c conda-forge      # if not already installed
#   bash scripts/download_av2.sh
#
# Tip: the full split is ~100 GB. You may Ctrl-C after a few hundred scenarios — the
# pipeline works on partial downloads.
set -euo pipefail

export DATASET_NAME="motion-forecasting"
# Target directory on your machine (override by exporting TARGET_DIR before running).
export TARGET_DIR="${TARGET_DIR:-$HOME/data/datasets/${DATASET_NAME}}"

mkdir -p "${TARGET_DIR}"

if ! command -v s5cmd >/dev/null 2>&1; then
  echo "ERROR: s5cmd not found. Install it with:  conda install s5cmd -c conda-forge" >&2
  exit 1
fi

echo ">> Downloading '${DATASET_NAME}' to ${TARGET_DIR}"
echo ">> (train/ val/ test/ subdirectories, each containing per-scenario folders)"

# The trailing /* copies the train, val, and test subtrees.
s5cmd --no-sign-request cp \
  "s3://argoverse/datasets/av2/${DATASET_NAME}/*" \
  "${TARGET_DIR}/"

echo ""
echo ">> Download complete. Point the code at it with:"
echo "     export FORESEE_DATA_ROOT=\"${TARGET_DIR}\""
