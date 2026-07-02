#!/usr/bin/env bash
# Launch the AI CTA Stroke Streamlit dashboard.
# Usage:  bash scripts/run_app.sh [--port 8501]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${2:-8501}"

# Find the conda env's Python / streamlit
CONDA_BIN=""
for p in \
    "$HOME/miniforge3/envs/laa-pipeline/bin" \
    "$HOME/miniconda3/envs/laa-pipeline/bin" \
    "$HOME/anaconda3/envs/laa-pipeline/bin"
do
    if [[ -x "${p}/streamlit" ]]; then
        CONDA_BIN="$p"
        break
    fi
done

if [[ -z "$CONDA_BIN" ]]; then
    echo "ERROR: laa-pipeline env not found or streamlit not installed."
    echo "Run:  bash scripts/setup_env.sh"
    exit 1
fi

export PATH="${CONDA_BIN}:$PATH"

URL="http://localhost:${PORT}"
echo "Starting UI at ${URL}"

# Open in the system default browser once the server is ready
(sleep 3 && \
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${URL}"          # Linux (opens Firefox, Chrome, etc.)
  elif command -v open >/dev/null 2>&1; then
    open "${URL}"              # macOS
  fi
) &

"${CONDA_BIN}/streamlit" run \
    "${REPO_ROOT}/app/streamlit_app.py" \
    --server.port "${PORT}" \
    --server.headless true \
    --browser.gatherUsageStats false
