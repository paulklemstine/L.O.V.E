#!/bin/bash
# love-run.sh — continuous posting on the fully local pipeline
# Usage: love-run.sh [extra love-cli.mjs flags]
#   Runs forever, one post at a time; love-cli.mjs owns the loop.
cd "$(dirname "$0")"
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
# flock alive in every child. Without 9>&- below, a `node` process outliving a
# Ctrl-C holds the lock for minutes after the script is gone, and the next
# launch is refused against a run that no longer exists. The child closes it.
#
# Deliberately not `exec`: in a pipeline, exec would replace the subshell
# running node and this script would fall straight through and exit, releasing
# the lock while node was still posting. As a plain pipeline the script blocks
# for the life of the process, so the lock is held for exactly as long as the run.
log "starting continuous mode (one post at a time, Ctrl-C to stop)"
node love-cli.mjs --post "$@" 9>&- 2>&1 | tee -a "$LOG" 9>&-
log "=== stopped (exit ${PIPESTATUS[0]}) ==="
