#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Optional virtualenv activation. Override with VENV_ACTIVATE if needed.
DEFAULT_VENV="/home/nataliehogg/Documents/Codes/myenvs/EverythingDev/venv/bin/activate"
if [[ -n "${VENV_ACTIVATE:-}" && -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
elif [[ -f "${DEFAULT_VENV}" ]]; then
  # shellcheck disable=SC1090
  source "${DEFAULT_VENV}"
fi

# Avoid thread oversubscription when running many tasks in parallel.
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Default sample sizes for QMC/MC (override via env).
export LOSCOV_NSAMP="${LOSCOV_NSAMP:-8192}"
export LOSCOV_CSAMP="${LOSCOV_CSAMP:-8192}"

# Number of QMC randomizations for error estimation (override via env).
# When NUMBA_NUM_THREADS=1, this ensures proper error computation.
# Higher values give better error estimates but take longer.
export LOSCOV_NUM_RANDOMIZATIONS="${LOSCOV_NUM_RANDOMIZATIONS:-8}"

# Data output root and run stamp for unique output folders.
export LOSCOV_DATA_ROOT="${LOSCOV_DATA_ROOT:-${ROOT_DIR}/data/my_results}"
export LOSCOV_RUN_STAMP="${LOSCOV_RUN_STAMP:-$(date +%y%m%d_%H%M)}"
mkdir -p "${LOSCOV_DATA_ROOT}"

echo "=== Precomputing correlations + tasks ==="
python -u correlations_and_distributions.py

TASK_FILE="${TASK_FILE:-${ROOT_DIR}/tasks.txt}"
if [[ ! -f "${TASK_FILE}" ]]; then
  echo "Task file not found: ${TASK_FILE}" >&2
  exit 1
fi

# Determine parallelism.
if command -v nproc >/dev/null 2>&1; then
  NPROC_DEFAULT="$(nproc)"
else
  NPROC_DEFAULT=16
fi
# Some environments report a single core; fall back to 16 for this machine.
if [[ "${NPROC_DEFAULT}" -lt 2 ]]; then
  NPROC_DEFAULT=16
fi
PARALLEL_JOBS="${PARALLEL_JOBS:-${NPROC_DEFAULT}}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/output_logs}"
mkdir -p "${LOG_DIR}"

echo "=== Running tasks with PARALLEL_JOBS=${PARALLEL_JOBS} ==="
start_time="$(date +%s)"

# Run tasks concurrently; output is interleaved in the log.
xargs -I {} -P "${PARALLEL_JOBS}" bash -lc "echo \"==> \$(date +%H:%M:%S) start: {}\"; python -u job.py {}; echo \"<== \$(date +%H:%M:%S) done: {}\"" < "${TASK_FILE}" \
  2>&1 | tee "${LOG_DIR}/parallel_jobs.log"

end_time="$(date +%s)"
runtime_seconds="$((end_time - start_time))"
runtime_minutes="$((runtime_seconds / 60))"
summary_line="=== Done in ${runtime_minutes} min (${runtime_seconds} sec) ==="
echo "${summary_line}"
echo "${summary_line}" >> "${LOG_DIR}/parallel_jobs.log"
