// Failure backoff for the run-forever post loop.
//
// The loop has no pause between successful posts -- the ~6 minute SDXL
// render is the natural spacing. But without any delay on failure, two fast
// failures in a row (bad credentials, a throw before the first network call)
// retry immediately and hammer Ollama in a tight loop. Successive failures
// back off exponentially up to a ceiling; any success resets the streak, so a
// long healthy run is never slowed down by failures that happened long ago.

export const BACKOFF_BASE_MS = 60_000;   // first failure waits 1 minute
export const BACKOFF_MAX_MS = 15 * 60_000; // ceiling: 15 minutes

// Delay before retry number `failures` (1 = first failure). 0 failures -> 0.
export function backoffDelay(failures, base = BACKOFF_BASE_MS, max = BACKOFF_MAX_MS) {
    if (failures <= 0) return 0;
    return Math.min(base * 2 ** (failures - 1), max);
}

// Stateful streak counter. fail() records a failure; reset() is called after
// any successful post so the next failure waits the base delay again.
export class Backoff {
    constructor(base = BACKOFF_BASE_MS, max = BACKOFF_MAX_MS) {
        this.base = base;
        this.max = max;
        this.failures = 0;
    }

    fail() {
        this.failures++;
        return this.delay;
    }

    reset() {
        this.failures = 0;
    }

    get delay() {
        return backoffDelay(this.failures, this.base, this.max);
    }
}
