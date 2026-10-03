#!/usr/bin/env python3
"""Push clean page cache out of RAM without root: allocate (and touch) anonymous memory up to
MemAvailable - KEEP_GIB, then exit and free it. The kernel drops clean cache (e.g. root-owned
docker layer files that posix_fadvise cannot reach) to satisfy the allocation. Only run while
no model is loaded. Prints MemFree before/after."""
import sys

KEEP_GIB = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0   # also runs as `python3 - < this file` (6 GiB)


def mem():
    d = {}
    for line in open('/proc/meminfo'):
        k, v = line.split(':', 1)
        d[k] = int(v.split()[0]) * 1024
    return d


m = mem()
target = m['MemAvailable'] - int(KEEP_GIB * 2**30)
before = m['MemFree'] / 2**30
chunks, got, step = [], 0, 2**30
while got + step <= target:
    chunks.append(b'\x01' * step)          # filled: every page is touched
    got += step
    if mem()['MemAvailable'] < KEEP_GIB * 2**30:
        break
del chunks
print(f'balloon: {got / 2**30:.0f} GiB touched; MemFree {before:.1f} -> {mem()["MemFree"] / 2**30:.1f} GiB')
