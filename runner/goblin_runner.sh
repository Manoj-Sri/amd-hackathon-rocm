#!/usr/bin/env bash
# goblin_runner.sh — wrap the user's training command with rocprofv3 + amd-smi.
#
# Architecture: brainstorming/architecture.md §5 (Profiling Pipeline).
#
# Inputs (env vars):
#   USER_SCRIPT     Path to the workload python file. Required.
#   OUT_DIR         Directory to write trace.csv / torch_profile.json /
#                   amd_smi.csv. Required. Created if missing.
#   STEPS           --max_steps argument forwarded to the user script. Default 10.
#   GOBLIN_GPU_ID   ROCR_VISIBLE_DEVICES value. Default 0.
#
# Outputs (in $OUT_DIR):
#   trace.csv          rocprofv3 kernel trace
#   torch_profile.json torch.profiler chrome trace (the user script writes this)
#   amd_smi.csv        amd-smi telemetry sampled at 200 ms
#   stdout.log         user-script stdout
#   stderr.log         user-script stderr
#
# Failure mode: any non-zero rocprofv3 exit short-circuits the script. On
# failure we dump the captured stdout/stderr logs to THIS script's own stderr
# so `subprocess.run(capture_output=True)` in LiveRunner sees the real error
# (not just an empty `[]` tail). LiveRunner then archives the whole OUT_DIR
# under bench_cache/last_runner_failure_<ts>/ so you can inspect after-the-fact.

set -uo pipefail

: "${USER_SCRIPT:?USER_SCRIPT env var is required}"
: "${OUT_DIR:?OUT_DIR env var is required}"
STEPS="${STEPS:-10}"

mkdir -p "$OUT_DIR"

# Pin to a single MI300X so concurrent benchmark runs don't fight.
export ROCR_VISIBLE_DEVICES="${GOBLIN_GPU_ID:-0}"

# Background HBM/power telemetry. Sample at 200 ms and stop on EXIT.
amd-smi monitor --csv --interval 0.2 \
    > "$OUT_DIR/amd_smi.csv" 2> "$OUT_DIR/amd_smi.err" &
AMD_SMI_PID=$!

cleanup() {
    if kill -0 "$AMD_SMI_PID" 2>/dev/null; then
        kill "$AMD_SMI_PID" 2>/dev/null || true
        wait "$AMD_SMI_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Dump the captured logs to *our* stderr on a non-zero rocprofv3 exit so the
# Python subprocess that spawned us actually sees the real error message.
# Without this the redirected stdout.log / stderr.log live inside the tempdir
# only and LiveRunner's stderr-tail check sees nothing.
dump_failure_logs() {
    local code=$?
    if [[ $code -ne 0 ]]; then
        {
            echo "=== goblin_runner.sh failed with exit code $code ==="
            echo "=== USER_SCRIPT: $USER_SCRIPT ==="
            echo "=== OUT_DIR:     $OUT_DIR ==="
            echo "=== ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-unset} ==="
            echo
            echo "=== last 50 lines of $OUT_DIR/stdout.log ==="
            tail -n 50 "$OUT_DIR/stdout.log" 2>/dev/null || echo "(stdout.log missing)"
            echo
            echo "=== last 50 lines of $OUT_DIR/stderr.log ==="
            tail -n 50 "$OUT_DIR/stderr.log" 2>/dev/null || echo "(stderr.log missing)"
            echo
            echo "=== last 20 lines of $OUT_DIR/amd_smi.err ==="
            tail -n 20 "$OUT_DIR/amd_smi.err" 2>/dev/null || echo "(amd_smi.err missing)"
        } 1>&2
    fi
    return $code
}

# rocprofv3 collects HSA + kernel traces. The user script is responsible for
# writing torch_profile.json (the agent injects torch.profiler around the
# training loop in Phase 3). --output-format csv keeps parsing simple.
set +e
rocprofv3 \
    --hsa-trace --kernel-trace \
    --output-directory "$OUT_DIR" \
    --output-file trace \
    --output-format csv \
    -- \
    python "$USER_SCRIPT" \
        --max_steps="$STEPS" \
        --torch_profile_out="$OUT_DIR/torch_profile.json" \
        > "$OUT_DIR/stdout.log" 2> "$OUT_DIR/stderr.log"
ROCPROF_EXIT=$?
set -e

if [[ $ROCPROF_EXIT -ne 0 ]]; then
    (exit $ROCPROF_EXIT) || dump_failure_logs
    exit $ROCPROF_EXIT
fi

# rocprofv3 may write trace_kernel_trace.csv etc. — normalize to trace.csv so
# profile_parser has one stable filename to look for.
if [[ ! -f "$OUT_DIR/trace.csv" ]]; then
    for candidate in "$OUT_DIR"/trace*kernel*.csv "$OUT_DIR"/*kernel_trace.csv; do
        if [[ -f "$candidate" ]]; then
            cp "$candidate" "$OUT_DIR/trace.csv"
            break
        fi
    done
fi

exit 0
