#!/bin/bash
# 4 张卡并行跑 4 个种子
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG="./exps/sdadapter_inr.json"
SEEDS=(1995 1996 1997 1998)
GPUS=(0 1 2 3)
LOGFILE="InR_adapter.log"

# 清空日志
> "$LOGFILE"

PIDS=()
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    GPU="${GPUS[$i]}"

    # 生成单种子临时 config，checkpoint 目录按种子隔离
    TMP_CFG="/tmp/sdadapter_inr_seed${SEED}.json"
    python3 -c "
import json
cfg = json.load(open('${CONFIG}'))
cfg['seed'] = [${SEED}]
cfg['filepath'] = cfg['filepath'].rstrip('/') + '_s${SEED}/'
json.dump(cfg, open('${TMP_CFG}', 'w'), indent=2)
"

    echo "[GPU ${GPU}] seed=${SEED} started"
    CUDA_VISIBLE_DEVICES=${GPU} python3 main.py --config="${TMP_CFG}" \
        >> "$LOGFILE" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} jobs: PIDs=${PIDS[*]}"
echo "Logs -> ${LOGFILE}"

# 等待全部完成
FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[DONE] seed=${SEEDS[$i]} (PID ${PIDS[$i]})"
    else
        echo "[FAIL] seed=${SEEDS[$i]} (PID ${PIDS[$i]}) exit=$?"
        FAIL=$((FAIL + 1))
    fi
done

# 汇总结果
echo ""
python3 calc_acc.py "$LOGFILE" 2>/dev/null || echo "(no complete matrix found yet)"

[ $FAIL -eq 0 ] && echo "All done." || echo "${FAIL} job(s) failed."
