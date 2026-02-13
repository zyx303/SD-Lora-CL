#!/bin/bash
# ============================================================================
# Adapter LR search: 4 GPUs in parallel, each running a different learning rate.
# Tests both the fixed adapter (no CA) and adapter+CA variants.
# Uses seed 1996 (stable seed) for quick comparison.
# ============================================================================

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SEED=1995
EPOCHS=20

# Learning rates to search (4 values = 4 GPUs)
LRS=(0.001 0.005 0.01 0.02)
GPUS=(0 1 2 3)

SEARCH_DIR="./search_adapter_lr"
LOG_DIR="${SEARCH_DIR}/logs"
CFG_DIR="${SEARCH_DIR}/configs"
mkdir -p "$LOG_DIR" "$CFG_DIR"

# ── Generate config files and launch jobs ──────────────────────────────────────
PIDS=()

for i in "${!LRS[@]}"; do
    LR="${LRS[$i]}"
    GPU="${GPUS[$i]}"
    TAG="lr${LR}"

    CKPT_DIR="${SEARCH_DIR}/ckpt_${TAG}/"
    mkdir -p "$CKPT_DIR"

    # Generate config JSON
    CFG="${CFG_DIR}/adapter_${TAG}.json"
    cat > "$CFG" <<EOF
{
    "prefix": "search",
    "dataset": "imagenetr",
    "memory_size": 0,
    "memory_per_class": 0,
    "fixed_memory": false,
    "shuffle": true,
    "init_cls": 20,
    "increment": 20,
    "model_name": "sdadapter",
    "backbone_type": "vit_base_patch16_224_sdadapter",
    "adapter_rank": 64,
    "device": ["0"],
    "optimizer": "adam",
    "scheduler": "constant",
    "filepath": "${CKPT_DIR}",
    "seed": [${SEED}],
    "init_epoch": ${EPOCHS},
    "init_lr": ${LR},
    "init_milestones": [10000],
    "init_lr_decay": 0,
    "init_weight_decay": 0.0005,
    "epochs": ${EPOCHS},
    "lrate": ${LR},
    "milestones": [10000],
    "lrate_decay": 0,
    "batch_size": 128,
    "weight_decay": 0.0002,
    "per_layer_scaling": false
}
EOF

    LOGFILE="${LOG_DIR}/adapter_${TAG}.log"
    echo "[GPU ${GPU}] LR=${LR} -> ${LOGFILE}"

    CUDA_VISIBLE_DEVICES=${GPU} python3 main.py --config="${CFG}" \
        > "${LOGFILE}" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} jobs: PIDs = ${PIDS[*]}"
echo "Waiting for all jobs to finish..."

# ── Wait and collect results ───────────────────────────────────────────────────
FAIL=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    LR="${LRS[$i]}"
    if wait "$PID"; then
        echo "[DONE] LR=${LR} (PID ${PID}) succeeded"
    else
        echo "[FAIL] LR=${LR} (PID ${PID}) failed with exit code $?"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "========================================"
echo "  Adapter LR Search Results"
echo "========================================"

for i in "${!LRS[@]}"; do
    LR="${LRS[$i]}"
    LOGFILE="${LOG_DIR}/adapter_${LR}.log"
    # Try to use calc_acc.py, fallback to grep
    TAG="lr${LR}"
    LOGFILE="${LOG_DIR}/adapter_${TAG}.log"
    if [ -f "$LOGFILE" ]; then
        echo ""
        echo "--- LR=${LR} ---"
        python3 calc_acc.py "$LOGFILE" 2>/dev/null || echo "  (no complete matrix found)"
    fi
done

echo ""
if [ $FAIL -eq 0 ]; then
    echo "All jobs completed successfully."
else
    echo "${FAIL} job(s) failed. Check logs in ${LOG_DIR}/"
fi
