#!/bin/bash
# love-run.sh — batch posting on the fully local pipeline
# Usage: love-run.sh [batches] [batch_size]
cd "$(dirname "$0")"
BATCHES="${1:-3}"
SIZE="${2:-10}"
LOG="love-run.log"

for i in $(seq 1 "$BATCHES"); do
  echo "[$(date '+%F %T')] === batch $i/$BATCHES ($SIZE posts) ===" >> "$LOG"
  node love-cli.mjs --batch "$SIZE" --post >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] === batch $i done (exit $?) ===" >> "$LOG"
done
echo "[$(date '+%F %T')] all $BATCHES batches complete" >> "$LOG"
