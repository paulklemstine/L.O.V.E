## 2024-05-24 - Parallel Requests
**Learning:** Sequential network requests are a major bottleneck. Parallelizing them using `concurrent.futures.ThreadPoolExecutor` provides massive speedups (1.5s -> 0.5s for 3 concurrent mock requests).
**Action:** Always look for opportunities to parallelize I/O bound operations, especially when fetching data from multiple sources.
## 2024-05-24 - Parallel Requests
**Learning:** Sequential network requests are a major bottleneck. Parallelizing them using `concurrent.futures.ThreadPoolExecutor` provides massive speedups (1.5s -> 0.5s for 3 concurrent mock requests).
**Action:** Always look for opportunities to parallelize I/O bound operations, especially when fetching data from multiple sources.
