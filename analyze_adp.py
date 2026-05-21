#!/usr/bin/env python3
from pathlib import Path

def read_varint(data, pos):
    value = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7

def aig_stats(data):
    newline = data.index(b'\n')
    header = data[:newline].decode('ascii').split()
    _max_var, inputs, latches, outputs, and_count = map(int, header[1:6])
    pos = newline + 1
    output_literals = []
    for _ in range(outputs):
        end = data.index(b'\n', pos)
        output_literals.append(int(data[pos:end]))
        pos = end + 1
    ands = []
    for idx in range(1, and_count + 1):
        lhs = 2 * (inputs + idx)
        delta0, pos = read_varint(data, pos)
        delta1, pos = read_varint(data, pos)
        rhs0 = lhs - delta0
        rhs1 = rhs0 - delta1
        ands.append((lhs, rhs0, rhs1))
    depth = {0: 0}
    for idx in range(inputs):
        depth[2 * (idx + 1)] = 0
    for lhs, rhs0, rhs1 in ands:
        depth[lhs] = max(depth[rhs0 & ~1], depth[rhs1 & ~1]) + 1
    delay = max((depth[lit & ~1] for lit in output_literals), default=0)
    area = len(ands)
    return inputs, area, delay, area * delay

results = []
for f in sorted(Path('output').glob('ex*.aig')):
    data = f.read_bytes()
    ni, a, d, adp = aig_stats(data)
    results.append((f.stem, ni, a, d, adp))

results.sort(key=lambda x: -x[4])
print('Top 30 worst ADP:')
for name, ni, a, d, adp in results[:30]:
    print(f'  {name}: inputs={ni}, area={a}, delay={d}, adp={adp}')
total = sum(r[4] for r in results)
print(f'Total ADP: {total}')
print(f'Total cases: {len(results)}')

print('\nAll cases sorted by ADP:')
for name, ni, a, d, adp in results:
    print(f'  {name}: inputs={ni}, area={a}, delay={d}, adp={adp}')
