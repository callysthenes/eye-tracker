#!/bin/bash
# Eye Tracker quick launch script
# Usage: ./run.sh [options]
#   --minimized   Start with minimized GUI
#   --camera N    Use webcam device N (default: 0)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Qt platform plugin path is set inside main.py before any imports.
# No environment overrides needed.

./venv/bin/python main.py "$@"
