#!/usr/bin/env bash
# Start the MONAILabel VISTA3D server for the LAA pilot.
#
# VISTA3D (interactive point/class prompting) serves the pilot CTAs in
# outputs/laa_pilot/monai_studies/. Connect from 3D Slicer via the official
# MONAILabel extension (Server: http://localhost:8000).
#
# Notes / gotchas (see memory laa-candidate-pipeline-env):
#   - Runs in the isolated conda env `monailabel` (python 3.10), invoked by
#     ABSOLUTE path with a clean PATH so the repo's .venv_dt does not shadow it.
#   - LAA = VISTA3D class 108 ("left atrial appendage").
#   - vista3d bundle (832 MB) lives under the app's model/ dir.
set -euo pipefail

REPO="/Users/sebastianfridman/Documents/pwd/AI_CTA_Stroke"
MENV=/opt/anaconda3/envs/monailabel
APP="$REPO/outputs/laa_pilot/monai_apps/monaibundle"
STUDIES="$REPO/outputs/laa_pilot/monai_studies"
PORT="${1:-8000}"

# free the port if a previous server is running
pkill -f "monailabel.main start_server" 2>/dev/null || true
lsof -ti "tcp:${PORT}" 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

exec env -i HOME="$HOME" PATH="$MENV/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  "$MENV/bin/python" -m monailabel.main start_server \
  --app "$APP" --studies "$STUDIES" \
  --conf models vista3d --conf preload false --port "$PORT"
