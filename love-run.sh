#!/bin/bash
# love-run.sh — continuous batch posting on the fully local pipeline
# Usage: love-run.sh [batches] [batch_size]
#   No [batches] argument -> runs continuously (loop forever)
cd "$(dirname "$0")"
BATCHES="${1:-0}"
SIZE="${2:-10}"
LOG="love-run.log"

run_batch () {
  local i="$1"
  echo "[$(date '+%F %T')] === batch $i ($SIZE posts) ===" >> "$LOG"
  node love-cli.mjs --batch "$SIZE" --post >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] === batch $i done (exit $?) ===" >> "$LOG"
}

if [ "$BATCHES" -gt 0 ] 2>/dev/null; then
  for i in $(seq 1 "$BATCHES"); do
    run_batch "$i/$BATCHES"
  done
  echo "[$(date '+%F %T')] all $BATCHES batches complete" >> "$LOG"
else
  echo "[$(date '+%F %T')] starting continuous mode ($SIZE posts per batch, Ctrl-C to stop)" >> "$LOG"
  i=1
  while true; do
    run_batch "$i"
    i=$((i + 1))
    sleep 60
  done
fi
