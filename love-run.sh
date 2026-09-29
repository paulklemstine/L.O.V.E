#!/bin/bash
# love-run.sh — continuous batch posting on the fully local pipeline
# Usage: love-run.sh [batches] [batch_size]
#   No [batches] argument -> runs continuously (loop forever)
cd "$(dirname "$0")"
BATCHES="${1:-0}"
SIZE="${2:-10}"
LOG="love-run.log"
LOCK="$PWD/.love-run.lock"

log () {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

# Single-instance guard. Two runs at once is not a cosmetic problem: the 6GB
# card cannot hold an SDXL render and a llama.cpp compute graph at the same
# time, and the loser is whichever asks for VRAM second -- that surfaces as
# "failed to allocate Vulkan0 buffer" and an Ollama 500. On 2026-09-28 a
# second launch collided with a running batch and both interleaved their
# output into this log.
#
# flock rather than a PID file: the kernel drops the lock when the holder
# dies, so a crashed or SIGKILLed run cannot wedge the script forever.
#
# Opened with >> not >, so a refused launch does not truncate the file and
# lose the running pid it is about to report. Truncating is safe *after* the
# lock is ours.
exec 9>>"$LOCK"
if ! flock -n 9; then
  log "refusing to start: love-run.sh is already running (pid $(cat "$LOCK" 2>/dev/null || echo '?'))"
  exit 1
fi
echo $$ > "$LOCK"
# Deliberately not unlinking the lock file on exit. flock locks the inode, not
# the path, so removing the file lets a second launcher lock the orphaned
# inode while a third creates a fresh file and locks that too -- two runs, two
# "held" locks. A leftover file is harmless: flock -n is the actual test, and
# the pid inside is only read to phrase the refusal.

# A lock held only by this shell is the point: an inherited fd would keep the
# flock alive in every child. Without 9>&- below, a `sleep 60` outliving a
# Ctrl-C (or `node` mid-render) holds the lock for minutes after the script
# is gone, and the next launch is refused against a run that no longer exists.
# Every child spawned below closes it explicitly.

run_batch () {
  local i="$1"
  log "=== batch $i ($SIZE posts) ==="
  node love-cli.mjs --batch "$SIZE" --post 9>&- 2>&1 | tee -a "$LOG" 9>&-
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
    sleep 60 9>&-
  done
fi
