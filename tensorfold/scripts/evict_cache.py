#!/usr/bin/env python3
"""Evict clean page cache of large files under the given directories, without root.

posix_fadvise(DONTNEED) on every regular file >= 16 MiB. Reads and writes nothing; only drops clean cached pages.
On GB10 unified memory, page cache competes with CUDA allocations at load time.
usage: evict_cache.py DIR [DIR ...]
"""
import os
import sys


def memfree_gib():
    return next(int(l.split()[1]) // 1048576 for l in open('/proc/meminfo') if l.startswith('MemFree:'))


before = memfree_gib()
n = 0
for root in sys.argv[1:]:
    for dp, _, fn in os.walk(os.path.expanduser(root), followlinks=False):
        for f in fn:
            p = os.path.join(dp, f)
            try:
                if os.path.islink(p) or os.path.getsize(p) < 16 << 20:
                    continue
                fd = os.open(p, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
                n += 1
            except OSError:
                pass
print(f'{os.uname().nodename}: fadvised {n} files; MemFree {before} -> {memfree_gib()} GiB')
