#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# GDS vs non-GDS safetensors loading benchmark
#
# Compares three modes:
#   1. Standard mmap path      (use_gds=False)
#   2. kvikio compat/POSIX     (use_gds=True, KVIKIO_COMPAT_MODE=AUTO)
#   3. kvikio with real GDS    (use_gds=True, KVIKIO_COMPAT_MODE=OFF)
#
# Each test is run RUNS times.  The first run is treated as warmup (page-cache
# fill / JIT overhead) and excluded from the reported statistics.
# =============================================================================

RUNS=${RUNS:-4}            # total runs per (mode × file); first is warmup
DEVICE=${DEVICE:-cuda:0}
DROP_CACHES=${DROP_CACHES:-1}  # set to 0 to skip cache dropping

# ── file list ────────────────────────────────────────────────────────────────
FILES=(
    "/home/work/gpt-oss-120b/model-00000-of-00014.safetensors"
    "/home/work/gemma-4-31B-it/model-00002-of-00002.safetensors"
)

# ── colours (disabled when piped) ────────────────────────────────────────────
if [ -t 1 ]; then
    BOLD="\033[1m" DIM="\033[2m" RST="\033[0m"
    GREEN="\033[32m" CYAN="\033[36m" YELLOW="\033[33m"
else
    BOLD="" DIM="" RST="" GREEN="" CYAN="" YELLOW=""
fi

# ── helpers ──────────────────────────────────────────────────────────────────
drop_caches() {
    if [ "$DROP_CACHES" = "1" ]; then
        sync
        echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
    fi
}

human_size() {
    local bytes=$1
    if (( bytes >= 1073741824 )); then
        printf "%.1f GB" "$(echo "scale=1; $bytes / 1073741824" | bc)"
    elif (( bytes >= 1048576 )); then
        printf "%.1f MB" "$(echo "scale=1; $bytes / 1048576" | bc)"
    else
        printf "%d B" "$bytes"
    fi
}

# ── benchmark python snippet ────────────────────────────────────────────────
# Prints: elapsed_sec num_tensors total_bytes
bench_py='
import sys, time, torch
from safetensors.torch import load_file

fpath    = sys.argv[1]
device   = sys.argv[2]
use_gds  = sys.argv[3] == "1"

torch.cuda.synchronize()
t0 = time.perf_counter()
tensors = load_file(fpath, device=device, use_gds=use_gds)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

total_bytes = sum(t.nelement() * t.element_size() for t in tensors.values())
print(f"{elapsed:.4f} {len(tensors)} {total_bytes}")
'

# ── run one benchmark ────────────────────────────────────────────────────────
# run_bench <label> <env_prefix> <use_gds_flag> <file>
run_bench() {
    local label="$1" env_prefix="$2" use_gds="$3" fpath="$4"
    local fname
    fname=$(basename "$fpath")
    local fsize
    fsize=$(stat -c%s "$fpath")

    printf "${BOLD}%-28s${RST} %-50s " "$label" "$fname"

    local times=()
    local num_tensors=0 total_bytes=0

    for ((i = 1; i <= RUNS; i++)); do
        drop_caches
        local result
        result=$(env $env_prefix python -c "$bench_py" "$fpath" "$DEVICE" "$use_gds" 2>&1)
        local elapsed ntensors tbytes
        read -r elapsed ntensors tbytes <<< "$result"
        num_tensors=$ntensors
        total_bytes=$tbytes

        if (( i == 1 )); then
            printf "${DIM}(warmup %.2fs)${RST} " "$elapsed"
        else
            times+=("$elapsed")
        fi
    done

    # compute min / avg / max
    local stats
    stats=$(python -c "
import sys
ts = [float(x) for x in sys.argv[1:]]
mn, mx, avg = min(ts), max(ts), sum(ts)/len(ts)
print(f'{mn:.3f} {avg:.3f} {mx:.3f}')
" "${times[@]}")

    local tmin tavg tmax
    read -r tmin tavg tmax <<< "$stats"

    local bw
    bw=$(python -c "print(f'{$total_bytes / $tavg / 1e9:.2f}')")

    printf "tensors=%-4d  size=%-8s  " "$num_tensors" "$(human_size "$total_bytes")"
    printf "min/avg/max = ${GREEN}%s${RST}/${CYAN}%s${RST}/${YELLOW}%s${RST} s  " "$tmin" "$tavg" "$tmax"
    printf "bw=${BOLD}%s GB/s${RST}\n" "$bw"
}

# ── main ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  safetensors GDS benchmark"
echo "  Device : $DEVICE"
echo "  Runs   : $RUNS (first is warmup)"
echo "  Files  : ${#FILES[@]}"
echo "============================================================"
echo ""

for fpath in "${FILES[@]}"; do
    printf "${BOLD}── %s (%s) ──${RST}\n" "$(basename "$fpath")" "$(human_size "$(stat -c%s "$fpath")")"

    run_bench "mmap (no GDS)"          ""                        0 "$fpath"
    run_bench "kvikio compat (AUTO)"   "KVIKIO_COMPAT_MODE=AUTO" 1 "$fpath"
    run_bench "kvikio GDS (OFF)"       "KVIKIO_COMPAT_MODE=OFF"  1 "$fpath"

    echo ""
done

echo "Done."
