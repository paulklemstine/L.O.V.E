#!/bin/bash
# love-run.sh — continuous batch posting on the fully local pipeline
# Usage: love-run.sh [batches] [batch_size]
#   No [batches] argument -> runs continuously (loop forever)
cd "$(dirname "$0")"
BATCHES="${1:-0}"
SIZE="${2:-10}"
LOG="love-run.log"

log () {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

run_batch () {
  local i="$1"
  log "=== batch $i ($SIZE posts) ==="
  node love-cli.mjs --batch "$SIZE" --post 2>&1 | tee -a "$LOG"
  log "=== batch $i done (exit ${PIPESTATUS[0]}) ==="
}

if [ "$BATCHES" -gt 0 ] 2>/dev/null; then
  for i in $(seq 1 "$BATCHES"); do
    run_batch "$i/$BATCHES"
  done
  log "all $BATCHES batches complete"
else
  log "starting continuous mode ($SIZE posts per batch, Ctrl-C to stop)"
  i=1
  while true; do
    run_batch "$i"
    i=$((i + 1))
    sleep 60
  done
fi
