"""sass_tally.py -- brief-8 style opcode tally for a SASS kernel body.

Usage: python sass_tally.py <sass-file> <start-line> <end-line> <label>
Counts static instructions (/*pc*/ lines) in [start,end), prints total,
per-64-weight figure (64 weights/lane/K-row), and the opcode histogram.
"""
import re
import sys

path, a, b, label = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ops = {}
n = 0
with open(path, encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f, start=1):
        if i < a or i >= b:
            continue
        m = re.search(r"/\*[0-9a-f]{4}\*/\s+(?:@!?P\d\s+)?([A-Z][A-Z0-9.]+)", line)
        if not m:
            continue
        op = m.group(1).split(".")[0]
        ops[op] = ops.get(op, 0) + 1
        n += 1
print("%s: static instructions in lines [%d,%d): %d" % (label, a, b, n))
print("per 64 weights (one lane, one K-row): %.5f" % (n / 64.0))
for op in sorted(ops):
    print("  %-7s %5d" % (op, ops[op]))
