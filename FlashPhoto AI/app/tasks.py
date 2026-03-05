from concurrent.futures import ThreadPoolExecutor
import os

# Scale workers to available CPU cores (min 4, max 16).
# Each worker can run a GPU-backed face-extraction in parallel.
_workers = min(max(os.cpu_count() or 4, 4), 16)

executor = ThreadPoolExecutor(max_workers=_workers)
print(f"Background thread pool: {_workers} workers")
