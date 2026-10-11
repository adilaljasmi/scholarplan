#!/usr/bin/env bash
# Capture dashboard screenshots for evidence/screenshots using headless Chrome.
# Usage (dashboard must be running on :8501):  bash docs/capture_screenshots.sh <ui_run_id> <eval_replanning_run_id>
set -u
CHROME="${CHROME:-/c/Program Files/Google/Chrome/Application/chrome.exe}"
OUT="$(cygpath -w "$PWD")\\evidence\\screenshots"
RUN="$1"
REPLAN="$2"

shot() {  # name, query string, window size
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --window-size="$3" \
    --virtual-time-budget=25000 --screenshot="$OUT\\$1.png" "http://localhost:8501/?$2" 2>&1 | grep -o "bytes written.*png" || echo "FAILED $1"
}

shot 01_dashboard_trace     "run=$RUN&tab=trace"                1600,1000
shot 02_subtasks            "run=$RUN&tab=subtasks"             1600,800
shot 03_sources             "run=$RUN&tab=sources"              1600,1000
shot 04_claims_verdicts     "run=$RUN&tab=claims"               1600,1300
shot 05_report              "run=$RUN&tab=report"               1600,2200
shot 06_metrics             "run=$RUN&tab=metrics"              1600,900
shot 07_replanning_feedback "db=eval&run=$REPLAN&tab=feedback"  1600,900
shot 08_replanning_subtasks "db=eval&run=$REPLAN&tab=subtasks"  1600,800
shot 09_replanning_trace    "db=eval&run=$REPLAN&tab=trace"     1600,1400
