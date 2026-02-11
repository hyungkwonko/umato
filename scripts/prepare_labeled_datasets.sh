#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/hj-n/labeled-datasets.git"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/external/labeled-datasets"

mkdir -p "${REPO_ROOT}/external"

if [[ -d "${TARGET_DIR}/.git" ]]; then
  git -C "${TARGET_DIR}" remote set-url origin "${REPO_URL}"
  git -C "${TARGET_DIR}" fetch --depth 1 origin master
  git -C "${TARGET_DIR}" checkout -B master origin/master
else
  git clone --depth 1 --filter=blob:none --sparse "${REPO_URL}" "${TARGET_DIR}"
  git -C "${TARGET_DIR}" checkout master
fi

git -C "${TARGET_DIR}" sparse-checkout init --cone
git -C "${TARGET_DIR}" sparse-checkout set npy

echo "Prepared labeled-datasets at: ${TARGET_DIR}"
echo "Datasets root for tests: ${TARGET_DIR}"
