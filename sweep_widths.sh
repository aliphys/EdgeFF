#!/usr/bin/env bash
# Sweep widths for Main.py with depth=3
# For each width: train once (no analysis), then run analysis 5 times and tee logs.
# Optional:
#  - ANALYSIS=specialization|energy|both|none (default: both)
#  - RESUME=1 to skip completed items and reuse per-width model backups
#  - RUNS=N to control number of analysis repetitions (default: 5)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="python"

DEPTH=3
START_WIDTH=100
END_WIDTH=1000
STEP=100

# Optional controls
ANALYSIS="${ANALYSIS:-both}"
RESUME="${RESUME:-0}"
RUNS="${RUNS:-5}"

# Resolve analysis flags
analysis_flags=()
case "$ANALYSIS" in
  specialization)
    analysis_flags=(--run_specialization)
    ;;
  energy)
    analysis_flags=(--run_energy_analysis)
    ;;
  both)
    analysis_flags=(--run_specialization --run_energy_analysis)
    ;;
  none)
    analysis_flags=()
    ;;
  *)
    echo "Unknown ANALYSIS value: $ANALYSIS (use specialization|energy|both|none)" >&2
    exit 1
    ;;
esac

# Ensure model directory exists (Main/Train expect this path)
mkdir -p "$SCRIPT_DIR/model"
# Ensure logs directory exists
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

for ((w=START_WIDTH; w<=END_WIDTH; w+=STEP)); do
  MODEL_DIR="$SCRIPT_DIR/model"
  MODEL_FILE="$MODEL_DIR/temp_"
  MODEL_BACKUP="$MODEL_DIR/temp_width${w}"

  echo "============================================"
  echo "Width $w: Training (no analysis)"
  echo "============================================"
  # Training run: no specialization or energy analysis
  if [[ "$RESUME" = "1" && -f "$MODEL_BACKUP" ]]; then
    echo "Resume enabled and model backup exists for width ${w}; skipping training."
  else
    ${PYTHON_BIN} Main.py \
      --depth "${DEPTH}" \
      --width "${w}" \
      --train True
    # Save a per-width backup of the trained model for resume
    if [[ -f "$MODEL_FILE" ]]; then
      cp -f "$MODEL_FILE" "$MODEL_BACKUP"
    else
      echo "Warning: expected model file '$MODEL_FILE' not found after training." >&2
    fi
  fi

  echo "============================================"
  echo "Width $w: Analysis runs (${RUNS}x)"
  echo "============================================"
  if [[ "${#analysis_flags[@]}" -eq 0 ]]; then
    echo "ANALYSIS=none, skipping analysis runs for width ${w}."
  else
    # Ensure the correct width model is the active 'temp_' before analyses
    if [[ -f "$MODEL_BACKUP" ]]; then
      cp -f "$MODEL_BACKUP" "$MODEL_FILE"
    fi
    for run in $(seq 1 "$RUNS"); do
      log_file="$LOG_DIR/width${w}run${run}.log"
      if [[ "$RESUME" = "1" && -s "$log_file" ]]; then
        echo "-- Log exists, skipping analysis ${run}/${RUNS} for width ${w}: $(basename "$log_file")"
        continue
      fi
      echo "-- Analysis ${run}/${RUNS} for width ${w} -> $(basename "$log_file")"
      # Analysis run: use trained model, enable requested analyses
      # Training is disabled by default (no --train True here)
      ${PYTHON_BIN} Main.py \
        --depth "${DEPTH}" \
        --width "${w}" \
        "${analysis_flags[@]}" | tee "$log_file"
    done
  fi
done

echo "All width sweeps completed."