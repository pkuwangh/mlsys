#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=${MODEL_ROOT:-"$HOME/scratch/checkpoints/Wan-AI"}
LOG_DIR=${LOG_DIR:-"$SCRIPT_DIR/logs/$(date +%Y%m%d-%H%M%S)"}
PYTHON=${PYTHON:-python}
case ${OPERATION:-all} in
    all) operations=(encoder decoder) ;;
    encoder|decoder) operations=("$OPERATION") ;;
    *) printf 'OPERATION must be all, encoder or decoder\n' >&2; exit 2 ;;
esac
case ${POLICY:-all} in
    all) policies=(bf16 fp32 tf32) ;;
    bf16|fp32|tf32) policies=("$POLICY") ;;
    *) printf 'POLICY must be all, bf16, fp32 or tf32\n' >&2; exit 2 ;;
esac
mkdir -p -- "$(dirname -- "$LOG_DIR")"
mkdir -- "$LOG_DIR" || exit 1
printf 'model\toperation\tpolicy\texit_status\tlog\n' > "$LOG_DIR/summary.tsv"
failed=0

for operation in "${operations[@]}"; do
    for policy in "${policies[@]}"; do
        arithmetic=(--dtype bf16)
        if [[ $policy == fp32 ]]; then
            arithmetic=(--dtype fp32 --no-allow-tf32)
        elif [[ $policy == tf32 ]]; then
            arithmetic=(--dtype fp32 --allow-tf32)
        fi
        for model in Wan2.2-TI2V-5B-Diffusers Wan2.1-T2V-1.3B-Diffusers; do
            log="$LOG_DIR/$model-$operation-$policy.log"
            command=("$PYTHON" "$SCRIPT_DIR/bench_wan_vae_$operation.py" --model "$MODEL_ROOT/$model" "${arithmetic[@]}")
            printf 'Command:' > "$log"
            printf ' %q' "${command[@]}" >> "$log"
            printf '\n' >> "$log"
            "${command[@]}" 2>&1 | tee -a "$log"
            status=$?
            printf '\nExit status: %s\n' "$status" | tee -a "$log"
            printf '%s\t%s\t%s\t%s\t%s\n' "$model" "$operation" "$policy" "$status" "$log" >> "$LOG_DIR/summary.tsv"
            if (( status != 0 )); then
                failed=1
            fi
        done
    done
done
printf '\nRun summary: %s/summary.tsv\n' "$LOG_DIR"
exit "$failed"
