#!/bin/bash
set -e

# Load .env
if [ -f .env ]; then
  echo "Loading .env..."
  set -a
  source .env
  set +a
else
  echo ".env file not found!"
  exit 1
fi

# Activate venv
if [ -f .venv_core/bin/activate ]; then
  source .venv_core/bin/activate
else
  echo ".venv_core not found!"
  exit 1
fi

# Check env vars
if [ -z "$JULES_API_KEY" ]; then
  echo "Error: JULES_API_KEY is not set."
  exit 1
fi

if [ -z "$GITHUB_TOKEN" ]; then
  if [ -n "$GITHUB_PERSONAL_ACCESS_TOKEN" ]; then
    echo "Mapping GITHUB_PERSONAL_ACCESS_TOKEN to GITHUB_TOKEN..."
    export GITHUB_TOKEN="$GITHUB_PERSONAL_ACCESS_TOKEN"
  else
    echo "Error: GITHUB_TOKEN is not set."
    exit 1
  fi
fi

echo "Environment loaded. Running tests..."
pytest tests/test_jules_lifecycle.py -v -s
