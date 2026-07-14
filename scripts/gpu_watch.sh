#!/usr/bin/env bash
# Report GPU memory and the current training speed once, so a run that is
# thrashing shared system memory (the YOLOv8s batch-8 failure mode) is caught in
# its first minute instead of after two wasted hours.
cd "$(dirname "$0")/.."
LOG="${1:-outputs/training/chain_console.log}"

MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)
LINE=$(tail -c 400 "$LOG" | tr '\r' '\n' | grep -aE "[0-9]+/[0-9]+" | tail -1)
SPEED=$(echo "$LINE" | grep -oE "[0-9.]+(it/s|s/it)" | tail -1)

echo "GPU memory : $MEM"
echo "speed      : ${SPEED:-unknown}"
echo "line       : ${LINE:0:110}"

case "$SPEED" in
  *s/it)
    echo "VERDICT    : SLOW - seconds per iteration means the model is spilling"
    echo "             into shared system memory. Reduce batch size or abandon."
    ;;
  *it/s)
    echo "VERDICT    : OK - iterations per second, running from VRAM."
    ;;
  *)
    echo "VERDICT    : could not determine speed yet."
    ;;
esac
