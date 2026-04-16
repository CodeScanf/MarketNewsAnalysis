#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yml"
ENV_NAME="market-news-analysis"
RECREATE_ENV="${RECREATE_ENV:-0}"
ENV_NAME_SET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)
      ENV_FILE="${ROOT_DIR}/environment.cpu.yml"
      shift
      ;;
    --gpu)
      ENV_FILE="${ROOT_DIR}/environment.gpu.yml"
      shift
      ;;
    --recreate)
      RECREATE_ENV=1
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [env-name] [--cpu|--gpu] [--recreate]" >&2
      exit 1
      ;;
    *)
      if [[ "${ENV_NAME_SET}" == "1" ]]; then
        echo "Unexpected extra argument: $1" >&2
        echo "Usage: $0 [env-name] [--cpu|--gpu] [--recreate]" >&2
        exit 1
      fi
      ENV_NAME="$1"
      ENV_NAME_SET=1
      shift
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH" >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}" >&2
  exit 1
fi

echo "Using environment file: ${ENV_FILE}"
echo "Target environment name: ${ENV_NAME}"

if [[ "${RECREATE_ENV}" == "1" ]] && conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Removing existing conda environment before recreate..."
  conda env remove --name "${ENV_NAME}" --yes
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Updating existing conda environment..."
  conda env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune
else
  echo "Creating new conda environment..."
  conda env create --file "${ENV_FILE}" --name "${ENV_NAME}"
fi

echo "Installing project package in editable mode..."
conda run -n "${ENV_NAME}" python -m pip install --no-deps -e "${ROOT_DIR}"

echo "Installing spaCy English model..."
if ! conda run -n "${ENV_NAME}" python -m spacy download en_core_web_sm; then
  echo "Warning: en_core_web_sm download failed. The project will fall back to a blank spaCy pipeline." >&2
fi

echo
echo "Environment ready."
echo "Activate with:"
echo "  conda activate ${ENV_NAME}"
