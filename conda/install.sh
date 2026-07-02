#!/usr/bin/env bash
# One-shot environment creation for Foresee.
# Usage:  bash conda/install.sh && conda activate foresee
set -euo pipefail

ENV_NAME="foresee"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

echo ">> Creating conda environment '${ENV_NAME}' from environment.yml ..."
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo ">> Environment '${ENV_NAME}' already exists; updating."
  conda env update -n "${ENV_NAME}" -f "${HERE}/environment.yml" --prune
else
  conda env create -f "${HERE}/environment.yml"
fi

echo ">> Installing the foresee package (editable) into '${ENV_NAME}' ..."
conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}"

echo ""
echo ">> Done. Activate with:  conda activate ${ENV_NAME}"
echo ">> Sanity check:        python -m tests.smoke_test"
