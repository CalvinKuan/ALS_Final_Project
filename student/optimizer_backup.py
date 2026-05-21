#!/usr/bin/env python3
"""AIG optimizer for the ALS final project.

The optimizer builds several equivalent AIG candidates, verifies them with a
bit-parallel simulator, scores them by area-delay product, and keeps the best.

Candidate sources:
    1. existing output/exNNN.aig, used as a safe fallback;
    2. polarity-aware ANF/FPRM synthesis;
    3. ROBDD synthesis under fixed and problem-dependent variable orders;
    4. ABC optimization flows, when student/abc is executable;
    5. mockturtle post-optimization on the top candidate AIGs.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


EVEN16 = [0] * 65536
ODD16 = [0] * 65536
for _value in range(65536):
    _even = 0
    _odd = 0
    for _idx in range(8):
        _even |= ((_value >> (2 * _idx)) & 1) << _idx
        _odd |= ((_value >> (2 * _idx + 1)) & 1) << _idx
    EVEN16[_value] = _even
    ODD16[_value] = _odd


DEFAULT_ORDERS = ("natural", "reverse", "interleave", "byte_msb", "influence_desc")
QUICK_ORDERS = ("reverse", "byte_msb")
DEFAULT_ANF_TERM_CAP = 768


def popcount(value: int) -> int:
    bit_count = getattr(value, "bit_count", None)
    if bit_count is not None:
        return bit_count()
    return bin(value).count("1")


@dataclass(frozen=True)
class AigStats:
    area: int
    delay: int

    @property
    def adp(self) -> int:
        return self.area * self.delay


@dataclass
class TruthProblem:
    inputs: int
    outputs: list[int]

    @property
    def rows(self) -> int:
        return 1 << self.inputs

    @property
    def input_mask(self) -> int:
        return (1 << self.inputs) - 1


@dataclass
class Candidate:
    name: str
    stats: AigStats
    data: bytes | None = None
    path: Path | None = None

    def read_bytes(self) -> bytes:
        if self.data is not None:
            return self.data
        if self.path is None:
            raise ValueError("candidate has neither data nor path")
        return self.path.read_bytes()


# -----------------------------------------------------------------------------
# AIGER parsing, writing, simulation, and scoring
# -----------------------------------------------------------------------------

def read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7


def encode_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def parse_binary_aig(data: bytes) -> tuple[int, list[int], list[tuple[int, int, int]]]:
    newline = data.index(b"\n")
    header = data[:newline].decode("ascii").split()
    if not header or header[0] != "aig":
        raise ValueError(f"unsupported AIGER header {header[0] if header else '<empty>'!r}")
    if len(header) < 6:
        raise ValueError("truncated AIGER header")

    _max_var, inputs, latches, outputs, and_count = map(int, header[1:6])
    if latches != 0:
        raise ValueError("sequential AIGER files are not supported")

    pos = newline + 1
    output_literals: list[int] = []
    for _ in range(outputs):
        end = data.index(b"\n", pos)
        output_literals.append(int(data[pos:end]))
        pos = end + 1

    ands: list[tuple[int, int, int]] = []
    for idx in range(1, and_count + 1):
        lhs = 2 * (inputs + idx)
        delta0, pos = read_varint(data, pos)
        delta1, pos = read_varint(data, pos)
        rhs0 = lhs - delta0
        rhs1 = rhs0 - delta1
        ands.append((lhs, rhs0, rhs1))

    return inputs, output_literals, ands


def aig_stats(data: bytes) -> AigStats:
    inputs, outputs, ands = parse_binary_aig(data)
    depth = {0: 0}
    for idx in range(inputs):
        depth[2 * (idx + 1)] = 0

    for lhs, rhs0, rhs1 in ands:
        depth[lhs] = max(depth[rhs0 & ~1], depth[rhs1 & ~1]) + 1

    delay = max((depth[lit & ~1] for lit in outputs), default=0)
    return AigStats(area=len(ands), delay=delay)


def input_truth_vectors(inputs: int) -> dict[int, int]:
    values = {0: 0}
    rows = 1 << inputs
    for idx in range(inputs):
        vector = 0
        for row in range(rows):
            if (row >> idx) & 1:
                vector |= 1 << row
        values[2 * (idx + 1)] = vector
    return values


def simulate_aig(data: bytes) -> tuple[int, list[int]]:
    inputs, outputs, ands = parse_binary_aig(data)
    mask = (1 << (1 << inputs)) - 1
    values = input_truth_vectors(inputs)

    def literal_value(literal: int) -> int:
        base = literal & ~1
        value = values[base]
        return mask ^ value if literal & 1 else value

    for lhs, rhs0, rhs1 in ands:
        values[lhs] = literal_value(rhs0) & literal_value(rhs1)

    return inputs, [literal_value(literal) for literal in outputs]


def is_equivalent_by_simulation(data: bytes, problem: TruthProblem) -> bool:
    try:
        inputs, simulated_outputs = simulate_aig(data)
        return inputs == problem.inputs and simulated_outputs == problem.outputs
    except (KeyError, ValueError, IndexError):
        return False


# -----------------------------------------------------------------------------
# Truth table reading
# -----------------------------------------------------------------------------

def detect_num_inputs_from_line(line: str) -> int:
    length = len(line.strip())
    if length <= 0 or length & (length - 1):
        raise ValueError(f"truth-table line length must be a power of two, got {length}")
    return int(math.log2(length))


def line_to_truth_int(line: str) -> int:
    stripped = line.strip()
    value = 0
    for idx, char in enumerate(stripped[::-1]):
        if char == "1":
            value |= 1 << idx
        elif char != "0":
            raise ValueError(f"unsupported truth-table character {char!r}")
    return value


def read_truth_problem(path: Path) -> TruthProblem:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty truth table: {path}")

    inputs = detect_num_inputs_from_line(lines[0])
    expected_len = 1 << inputs
    for line in lines:
        if len(line) != expected_len:
            raise ValueError(
                f"inconsistent truth-table line length in {path.name}: "
                f"expected {expected_len}, got {len(line)}"
            )

    return TruthProblem(inputs=inputs, outputs=[line_to_truth_int(line) for line in lines])


# -----------------------------------------------------------------------------
# Variable orders and ROBDD synthesis
# -----------------------------------------------------------------------------

def variable_influences(problem: TruthProblem) -> list[int]:
    scores = [0] * problem.inputs
    for truth in problem.outputs:
        for bit in range(problem.inputs):
            step = 1 << bit
            jump = step << 1
            score = 0
            for base in range(0, problem.rows, jump):
                low = (truth >> base) & ((1 << step) - 1)
                high = (truth >> (base + step)) & ((1 << step) - 1)
                score += popcount(low ^ high)
            scores[bit] += score
    return scores


def make_order_library(problem: TruthProblem) -> dict[str, list[int]]:
    inputs = problem.inputs
    half = inputs // 2
    lower = list(range(half))
    upper = list(range(half, inputs))

    interleave: list[int] = []
    interleave_rev: list[int] = []
    for i in range(max(len(lower), len(upper))):
        if i < len(lower):
            interleave.append(lower[i])
        if i < len(upper):
            interleave.append(upper[i])
        if i < len(upper):
            interleave_rev.append(upper[i])
        if i < len(lower):
            interleave_rev.append(lower[i])

    if inputs == 16:
        byte_msb = [bit for pair in zip(range(7, -1, -1), range(15, 7, -1)) for bit in pair]
        byte_lsb = [bit for pair in zip(range(8), range(8, 16)) for bit in pair]
    else:
        byte_msb = list(reversed(range(inputs)))
        byte_lsb = list(range(inputs))

    influences = variable_influences(problem)
    influence_desc = sorted(range(inputs), key=lambda bit: (-influences[bit], bit))
    influence_asc = sorted(range(inputs), key=lambda bit: (influences[bit], bit))

    return {
        "natural": list(range(inputs)),
        "reverse": list(reversed(range(inputs))),
        "interleave": interleave,
        "interleave_rev": interleave_rev,
        "byte_msb": byte_msb,
        "byte_lsb": byte_lsb,
        "influence_desc": influence_desc,
        "influence_asc": influence_asc,
    }


def choose_orders(problem: TruthProblem, effort: str, requested: str | None) -> list[tuple[str, list[int]]]:
    order_library = make_order_library(problem)
    if requested:
        names = [name.strip() for name in requested.split(",") if name.strip()]
    elif effort == "quick":
        names = list(QUICK_ORDERS)
    elif effort == "high":
        names = list(order_library)
    else:
        names = list(DEFAULT_ORDERS)

    unknown = [name for name in names if name not in order_library]
    if unknown:
        raise ValueError(f"unknown BDD order(s): {', '.join(unknown)}")

    result: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for name in names:
        order = order_library[name]
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            result.append((name, order))
    return result


def reorder_truth(truth: int, inputs: int, permutation: list[int]) -> int:
    rows = 1 << inputs
    reordered = 0
    for new_idx in range(rows):
        old_idx = 0
        for level, variable in enumerate(permutation):
            if (new_idx >> level) & 1:
                old_idx |= 1 << variable
        if (truth >> old_idx) & 1:
            reordered |= 1 << new_idx
    return reordered


def split_low_high(truth: int, remaining_vars: int) -> tuple[int, int]:
    bit_count = 1 << remaining_vars
    chunks = (bit_count + 15) // 16
    low = 0
    high = 0
    out_shift = 0
    for chunk_idx in range(chunks):
        chunk = (truth >> (16 * chunk_idx)) & 0xFFFF
        low |= EVEN16[chunk] << out_shift
        high |= ODD16[chunk] << out_shift
        out_shift += 8
    return low, high


class BddManager:
    def __init__(self, order: list[int]) -> None:
        self.order = order
        self.nodes: list[tuple[int, int, int] | None] = [None, None]
        self.unique: dict[tuple[int, int, int], int] = {}
        self.cache: dict[tuple[int, int], int] = {}

    def build(self, level: int, remaining_vars: int, truth: int) -> int:
        if truth == 0:
            return 0
        if truth == (1 << (1 << remaining_vars)) - 1:
            return 1

        key = (level, truth)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        low_truth, high_truth = split_low_high(truth, remaining_vars)
        low = self.build(level + 1, remaining_vars - 1, low_truth)
        high = self.build(level + 1, remaining_vars - 1, high_truth)

        if low == high:
            result = low
        else:
            unique_key = (level, low, high)
            result = self.unique.get(unique_key)
            if result is None:
                result = len(self.nodes)
                self.nodes.append((level, low, high))
                self.unique[unique_key] = result

        self.cache[key] = result
        return result


# -----------------------------------------------------------------------------
# AIG construction helpers
# -----------------------------------------------------------------------------

class AigBuilder:
    def __init__(self, inputs: int, comment: str) -> None:
        self.inputs = inputs
        self.comment = comment
        self.ands: list[tuple[int, int, int]] = []
        self.hash: dict[tuple[int, int], int] = {}
        self.depth = {0: 0}
        for idx in range(inputs):
            self.depth[2 * (idx + 1)] = 0

    def literal_depth(self, literal: int) -> int:
        return self.depth[literal & ~1]

    def mk_and(self, lhs_lit: int, rhs_lit: int) -> int:
        if lhs_lit == 0 or rhs_lit == 0:
            return 0
        if lhs_lit == 1:
            return rhs_lit
        if rhs_lit == 1:
            return lhs_lit
        if lhs_lit == rhs_lit:
            return lhs_lit
        if lhs_lit == (rhs_lit ^ 1):
            return 0

        if lhs_lit < rhs_lit:
            lhs_lit, rhs_lit = rhs_lit, lhs_lit
        key = (lhs_lit, rhs_lit)
        existing = self.hash.get(key)
        if existing is not None:
            return existing

        literal = 2 * (self.inputs + len(self.ands) + 1)
        self.ands.append((literal, lhs_lit, rhs_lit))
        self.hash[key] = literal
        self.depth[literal] = max(self.literal_depth(lhs_lit), self.literal_depth(rhs_lit)) + 1
        return literal

    def mk_or(self, lhs_lit: int, rhs_lit: int) -> int:
        return self.mk_and(lhs_lit ^ 1, rhs_lit ^ 1) ^ 1

    def mk_xor(self, lhs_lit: int, rhs_lit: int) -> int:
        if lhs_lit == 0:
            return rhs_lit
        if rhs_lit == 0:
            return lhs_lit
        if lhs_lit == 1:
            return rhs_lit ^ 1
        if rhs_lit == 1:
            return lhs_lit ^ 1
        if lhs_lit == rhs_lit:
            return 0
        if lhs_lit == (rhs_lit ^ 1):
            return 1
        left = self.mk_and(lhs_lit, rhs_lit ^ 1)
        right = self.mk_and(lhs_lit ^ 1, rhs_lit)
        return self.mk_or(left, right)

    def mk_balanced_and(self, literals: list[int]) -> int:
        if not literals:
            return 1
        layer = literals[:]
        while len(layer) > 1:
            nxt: list[int] = []
            it = iter(layer)
            for lhs in it:
                rhs = next(it, None)
                nxt.append(lhs if rhs is None else self.mk_and(lhs, rhs))
            layer = nxt
        return layer[0]

    def mk_balanced_xor(self, literals: list[int]) -> int:
        if not literals:
            return 0
        layer = literals[:]
        while len(layer) > 1:
            nxt: list[int] = []
            it = iter(layer)
            for lhs in it:
                rhs = next(it, None)
                nxt.append(lhs if rhs is None else self.mk_xor(lhs, rhs))
            layer = nxt
        return layer[0]

    def mk_ite(self, condition: int, high: int, low: int) -> int:
        if high == low:
            return high
        if high == 1 and low == 0:
            return condition
        if high == 0 and low == 1:
            return condition ^ 1
        if high == 1:
            return self.mk_or(condition, low)
        if high == 0:
            return self.mk_and(condition ^ 1, low)
        if low == 1:
            return self.mk_or(condition ^ 1, high)
        if low == 0:
            return self.mk_and(condition, high)

        high_term = self.mk_and(condition, high)
        low_term = self.mk_and(condition ^ 1, low)
        return self.mk_or(high_term, low_term)

    def stats_for_outputs(self, outputs: list[int]) -> AigStats:
        delay = max((self.literal_depth(literal) for literal in outputs), default=0)
        return AigStats(area=len(self.ands), delay=delay)

    def to_binary_aig(self, outputs: list[int]) -> bytes:
        max_var = self.inputs + len(self.ands)
        header = f"aig {max_var} {self.inputs} 0 {len(outputs)} {len(self.ands)}\n"
        data = bytearray(header.encode("ascii"))
        for literal in outputs:
            data.extend(f"{literal}\n".encode("ascii"))

        for lhs, rhs0, rhs1 in self.ands:
            if rhs0 < rhs1:
                rhs0, rhs1 = rhs1, rhs0
            if not (lhs > rhs0 >= rhs1):
                raise ValueError("AIGER literal order invariant violated")
            data.extend(encode_varint(lhs - rhs0))
            data.extend(encode_varint(rhs0 - rhs1))

        data.extend(f"c\n{self.comment}\n".encode("ascii"))
        return bytes(data)


# -----------------------------------------------------------------------------
# Candidate generators
# -----------------------------------------------------------------------------

def synthesize_bdd_candidate(problem: TruthProblem, order_name: str, order: list[int]) -> Candidate:
    manager = BddManager(order)
    roots: list[int] = []
    for truth in problem.outputs:
        roots.append(manager.build(0, problem.inputs, reorder_truth(truth, problem.inputs, order)))

    builder = AigBuilder(problem.inputs, f"optimizer.py ROBDD order={order_name}")
    memo = {0: 0, 1: 1}

    def emit(node_id: int) -> int:
        cached = memo.get(node_id)
        if cached is not None:
            return cached
        node = manager.nodes[node_id]
        if node is None:
            raise ValueError("invalid BDD node")
        level, low, high = node
        condition = 2 * (order[level] + 1)
        literal = builder.mk_ite(condition, emit(high), emit(low))
        memo[node_id] = literal
        return literal

    output_literals = [emit(root) for root in roots]
    stats = builder.stats_for_outputs(output_literals)
    data = builder.to_binary_aig(output_literals)
    return Candidate(name=f"bdd:{order_name}", stats=stats, data=data)


def truth_to_anf_terms(truth: int, inputs: int, term_cap: int) -> list[int] | None:
    rows = 1 << inputs
    coeff = [(truth >> idx) & 1 for idx in range(rows)]
    for bit in range(inputs):
        step = 1 << bit
        jump = step << 1
        for base in range(0, rows, jump):
            for off in range(step):
                coeff[base + step + off] ^= coeff[base + off]

    terms: list[int] = []
    for mask, value in enumerate(coeff):
        if value:
            terms.append(mask)
            if len(terms) > term_cap:
                return None
    return terms


def phase_truth(truth: int, problem: TruthProblem, phase: int) -> int:
    if phase == 0:
        return truth
    transformed = 0
    for y_idx in range(problem.rows):
        if (truth >> (y_idx ^ phase)) & 1:
            transformed |= 1 << y_idx
    return transformed


def make_phase_library(problem: TruthProblem, effort: str, requested: str | None) -> list[tuple[str, int]]:
    named: dict[str, int] = {
        "none": 0,
        "all": problem.input_mask,
        "lower": (1 << (problem.inputs // 2)) - 1,
        "upper": problem.input_mask ^ ((1 << (problem.inputs // 2)) - 1),
        "alt0": sum(1 << bit for bit in range(0, problem.inputs, 2)),
        "alt1": sum(1 << bit for bit in range(1, problem.inputs, 2)),
    }

    if requested:
        names = [name.strip() for name in requested.split(",") if name.strip()]
    elif effort == "quick":
        names = ["none"]
    elif effort == "high":
        names = list(named)
    else:
        names = ["none", "all", "lower", "upper", "alt0", "alt1"]

    unknown = [name for name in names if name not in named]
    if unknown:
        raise ValueError(f"unknown ANF phase(s): {', '.join(unknown)}")

    result: list[tuple[str, int]] = []
    seen: set[int] = set()
    for name in names:
        phase = named[name]
        if phase not in seen:
            seen.add(phase)
            result.append((name, phase))
    return result


def synthesize_anf_candidate(
    problem: TruthProblem,
    phase_name: str,
    phase: int,
    term_cap: int,
) -> Candidate | None:
    output_terms: list[list[int]] = []
    unique_terms: set[int] = set()
    for truth in problem.outputs:
        terms = truth_to_anf_terms(phase_truth(truth, problem, phase), problem.inputs, term_cap)
        if terms is None:
            return None
        output_terms.append(terms)
        unique_terms.update(terms)
        if len(unique_terms) > term_cap * max(1, len(problem.outputs)):
            return None

    builder = AigBuilder(problem.inputs, f"optimizer.py ANF/FPRM phase={phase_name}")
    product_cache: dict[int, int] = {0: 1}

    def product(mask: int) -> int:
        cached = product_cache.get(mask)
        if cached is not None:
            return cached
        literals = [
            (2 * (idx + 1)) ^ ((phase >> idx) & 1)
            for idx in range(problem.inputs)
            if (mask >> idx) & 1
        ]
        literal = builder.mk_balanced_and(literals)
        product_cache[mask] = literal
        return literal

    outputs = [builder.mk_balanced_xor([product(mask) for mask in terms]) for terms in output_terms]
    stats = builder.stats_for_outputs(outputs)
    data = builder.to_binary_aig(outputs)
    return Candidate(name=f"anf:{phase_name}", stats=stats, data=data)


# -----------------------------------------------------------------------------
# ABC and mockturtle candidate generation
# -----------------------------------------------------------------------------

def command_path(path: Path) -> str:
    return path.as_posix()


ABC_FLOWS = {
    # ---- original flows ----
    "abc:baseline": "st",
    "abc:rw": "st; strash; balance; rewrite; rewrite -z; balance; refactor; refactor -z; balance",
    "abc:rwz": "st; strash; rewrite -z; refactor -z; balance; rewrite -z; balance",
    "abc:dc2": "st; strash; dc2; balance; rewrite -z; refactor -z; dc2; balance",
    "abc:dc2x": "st; strash; balance; dc2; rewrite; refactor; dc2; balance; rewrite -z; balance",
    "abc:rs": "st; strash; resub; resub -z; balance; rewrite -z; refactor -z; balance",
    "abc:rs2": "st; strash; rewrite -z; resub -z; refactor -z; resub -K 8; balance; dc2",
    "abc:if6": "st; strash; if -K 6; strash; balance; rewrite -z; refactor -z; balance",
    "abc:if8": "st; strash; if -K 8; strash; dc2; balance; rewrite -z; balance",
    "abc:aig": "st; strash; &get; &dc2; &put; balance; rewrite -z; refactor -z; balance",
    "abc:compress2": "st; strash; balance; rewrite; refactor; balance; rewrite -z; refactor -z; balance; dc2; balance",
    # ---- extended flows for ADP optimisation ----
    # multi-pass resyn (two full compress2 sweeps)
    "abc:resyn2": "st; strash; balance; rewrite; refactor; balance; rewrite -z; refactor -z; balance; rewrite; refactor; balance; rewrite -z; refactor -z; balance",
    # triple compress sweep with dc2
    "abc:compress3": "st; strash; balance; rewrite; refactor; balance; rewrite -z; refactor -z; balance; dc2; balance; rewrite; refactor; balance; rewrite -z; refactor -z; balance",
    # three dc2 passes before polishing
    "abc:dc2_3": "st; strash; dc2; dc2; dc2; balance; rewrite -z; refactor -z; balance",
    # resub with K=6 (finds more substitutions)
    "abc:rs_k6": "st; strash; resub -K 6; resub -K 6 -N 2; balance; rewrite -z; refactor -z; balance",
    # resub with K=8
    "abc:rs_k8": "st; strash; resub -K 8; resub -K 8 -N 2; balance; rewrite -z; refactor -z; balance",
    # interleaved resub/rewrite/refactor with different K
    "abc:rs3": "st; strash; balance; rewrite; resub -K 6; refactor; balance; rewrite -z; resub -K 8; refactor -z; balance",
    # fraig merging then polish
    "abc:fraig": "st; strash; fraig; balance; rewrite -z; refactor -z; balance",
    # lookahead mapping K=4 → recompose
    "abc:if4": "st; strash; if -K 4; strash; balance; rewrite -z; refactor -z; balance",
    # lookahead mapping K=5 → recompose
    "abc:if5": "st; strash; if -K 5; strash; balance; rewrite -z; refactor -z; balance",
    # two rounds of &dc2
    "abc:aig2": "st; strash; &get; &dc2; &dc2; &put; balance; rewrite -z; refactor -z; balance",
    # dc2 first for area, then rewrite -z for delay (ADP-focused)
    "abc:adp1": "st; strash; balance; rewrite; refactor; dc2; balance; rewrite -z; refactor -z; balance",
    # alternating dc2 and rewrite rounds (ADP-focused)
    "abc:adp2": "st; strash; dc2; balance; rewrite; refactor; dc2; balance; rewrite -z; refactor -z; dc2; balance",
    # long sequence: two full sweeps with dc2 in between
    "abc:long1": "st; strash; balance; rewrite; refactor; rewrite -z; refactor -z; balance; dc2; rewrite; refactor; rewrite -z; refactor -z; balance; dc2; balance",
}

# ABC optimisation flows that start from an existing AIG file rather than a truth table.
# These are used in the post-synthesis AIG-refinement stage.
AIG_FLOWS = {
    "aig:rw":     "strash; balance; rewrite; rewrite -z; balance; refactor; refactor -z; balance",
    "aig:dc2":    "strash; dc2; balance; rewrite -z; refactor -z; dc2; balance",
    "aig:resyn2": "strash; balance; rewrite; refactor; balance; rewrite -z; refactor -z; balance; rewrite; refactor; balance; rewrite -z; refactor -z; balance",
    "aig:rs":     "strash; resub; resub -z; balance; rewrite -z; refactor -z; balance",
    "aig:dc2x":   "strash; balance; dc2; rewrite; refactor; dc2; balance; rewrite -z; balance",
    "aig:adp":    "strash; balance; rewrite; refactor; dc2; balance; rewrite -z; refactor -z; balance",
    "aig:dc3":    "strash; dc2; dc2; dc2; balance; rewrite -z; refactor -z; balance",
}


def run_abc_flow(abc: Path, truth: Path, output: Path, flow: str, timeout: int) -> bool:
    command = (
        f"read_truth -xf {command_path(truth)}; "
        f"{flow}; "
        f"write_aiger -s {command_path(output)}"
    )
    try:
        result = subprocess.run(
            [str(abc), "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and output.is_file()


def executable_available(executable: Path, timeout: int, args: list[str] | None = None) -> bool:
    if not executable.is_file():
        return False
    try:
        result = subprocess.run(
            [str(executable)] + (args or []),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode in (0, 2)


def run_mockturtle_flow(
    mockturtle: Path,
    source: Candidate,
    tmp_root: Path,
    case_name: str,
    flow: str,
    timeout: int,
) -> Candidate | None:
    input_path = tmp_root / f"{case_name}_mockturtle_in.aig"
    output_path = tmp_root / f"{case_name}_mockturtle_{flow}.aig"
    input_path.write_bytes(source.read_bytes())

    try:
        result = subprocess.run(
            [str(mockturtle), str(input_path), str(output_path), flow],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0 or not output_path.is_file():
        return None

    try:
        data = output_path.read_bytes()
        stats = aig_stats(data)
    except (ValueError, IndexError):
        return None

    return Candidate(name=f"mt:{flow}", stats=stats, data=data)


def run_abc_aig_candidate(
    abc: Path,
    source: Candidate,
    tmp_root: Path,
    case_name: str,
    flow_name: str,
    flow: str,
    timeout: int,
) -> Candidate | None:
    """Run an ABC optimisation flow starting from an existing AIG (not a truth table)."""
    safe_flow = flow_name.replace(":", "_")
    input_path = tmp_root / f"{case_name}_aig_in.aig"
    output_path = tmp_root / f"{case_name}_{safe_flow}.aig"
    input_path.write_bytes(source.read_bytes())

    command = (
        f"read {command_path(input_path)}; "
        f"{flow}; "
        f"write_aiger -s {command_path(output_path)}"
    )
    try:
        result = subprocess.run(
            [str(abc), "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0 or not output_path.is_file():
        return None

    try:
        data = output_path.read_bytes()
        stats = aig_stats(data)
    except (ValueError, IndexError):
        return None

    return Candidate(name=flow_name, stats=stats, data=data)


# -----------------------------------------------------------------------------
# Candidate selection
# -----------------------------------------------------------------------------

def existing_candidate(path: Path, problem: TruthProblem, verify: bool) -> Candidate | None:
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
        parsed_inputs, _outputs, _ands = parse_binary_aig(data)
        if parsed_inputs != problem.inputs:
            return None
        stats = aig_stats(data)
    except (ValueError, IndexError):
        return None
    if verify and not is_equivalent_by_simulation(data, problem):
        return None
    return Candidate(name="existing", stats=stats, data=data)


def better(candidate: Candidate, incumbent: Candidate | None) -> bool:
    if incumbent is None:
        return True
    left = (candidate.stats.adp, candidate.stats.area, candidate.stats.delay)
    right = (incumbent.stats.adp, incumbent.stats.area, incumbent.stats.delay)
    return left < right


def optimize_case(
    truth: Path,
    output: Path,
    abc: Path,
    mockturtle: Path,
    order_items: list[tuple[str, list[int]]],
    phase_items: list[tuple[str, int]],
    timeout: int,
    use_abc: bool,
    use_mockturtle: bool,
    mockturtle_flows: list[str],
    mockturtle_top_k: int,
    mockturtle_rounds: int,
    keep_existing: bool,
    verify_existing: bool,
    anf_term_cap: int,
    abc_aig_top_k: int = 3,
    abc_aig_rounds: int = 1,
) -> Candidate:
    problem = read_truth_problem(truth)
    output.parent.mkdir(parents=True, exist_ok=True)
    best: Candidate | None = None
    candidates: list[Candidate] = []
    seen_candidate_hashes: set[bytes] = set()

    def add_candidate(candidate: Candidate | None) -> bool:
        nonlocal best
        if candidate is None:
            return False
        data = candidate.read_bytes()
        digest = hashlib.sha256(data).digest()
        if digest in seen_candidate_hashes:
            return False
        if not is_equivalent_by_simulation(data, problem):
            return False
        seen_candidate_hashes.add(digest)
        if candidate.data is None:
            candidate = Candidate(name=candidate.name, stats=candidate.stats, data=data)
        candidates.append(candidate)
        if better(candidate, best):
            best = candidate
        return True

    if keep_existing:
        add_candidate(existing_candidate(output, problem, verify_existing))

    for phase_name, phase in phase_items:
        add_candidate(synthesize_anf_candidate(problem, phase_name, phase, anf_term_cap))

    for order_name, order in order_items:
        try:
            add_candidate(synthesize_bdd_candidate(problem, order_name, order))
        except RecursionError:
            continue

    if use_abc:
        with tempfile.TemporaryDirectory(prefix=f"{truth.stem}_abc_", dir=output.parent) as tmp_dir:
            tmp_root = Path(tmp_dir)
            for flow_name, flow in ABC_FLOWS.items():
                tmp_output = tmp_root / f"{truth.stem}_{flow_name.replace(':', '_')}.aig"
                if not run_abc_flow(abc, truth, tmp_output, flow, timeout):
                    continue
                try:
                    data = tmp_output.read_bytes()
                    parsed_inputs, _outputs, _ands = parse_binary_aig(data)
                    if parsed_inputs != problem.inputs:
                        continue
                    stats = aig_stats(data)
                except (ValueError, IndexError):
                    continue
                add_candidate(Candidate(name=flow_name, stats=stats, data=data))

    if use_mockturtle and candidates:
        tried_mockturtle: set[tuple[bytes, str]] = set()
        with tempfile.TemporaryDirectory(prefix=f"{truth.stem}_mt_", dir=output.parent) as tmp_dir:
            tmp_root = Path(tmp_dir)
            for round_idx in range(mockturtle_rounds):
                sources = sorted(
                    candidates,
                    key=lambda candidate: (
                        candidate.stats.adp,
                        candidate.stats.area,
                        candidate.stats.delay,
                        candidate.name,
                    ),
                )[:mockturtle_top_k]
                added_this_round = 0
                for source_index, source in enumerate(sources):
                    source_digest = hashlib.sha256(source.read_bytes()).digest()
                    for flow in mockturtle_flows:
                        key = (source_digest, flow)
                        if key in tried_mockturtle:
                            continue
                        tried_mockturtle.add(key)
                        candidate = run_mockturtle_flow(
                            mockturtle=mockturtle,
                            source=source,
                            tmp_root=tmp_root,
                            case_name=f"{truth.stem}_{round_idx}_{source_index}",
                            flow=flow,
                            timeout=timeout,
                        )
                        if add_candidate(candidate):
                            added_this_round += 1
                if added_this_round == 0:
                    break

    # ABC AIG refinement: run AIG-to-AIG ABC flows on the top-k candidates found so
    # far (including mockturtle results).  Different starting points lead ABC to
    # different local optima, often improving on the truth-table starting point.
    if use_abc and candidates and abc_aig_rounds > 0:
        tried_aig: set[tuple[bytes, str]] = set()
        with tempfile.TemporaryDirectory(prefix=f"{truth.stem}_abc_aig_", dir=output.parent) as tmp_dir:
            tmp_root = Path(tmp_dir)
            for round_idx in range(abc_aig_rounds):
                sources = sorted(
                    candidates,
                    key=lambda c: (c.stats.adp, c.stats.area, c.stats.delay, c.name),
                )[:abc_aig_top_k]
                added_this_round = 0
                for src_idx, source in enumerate(sources):
                    src_digest = hashlib.sha256(source.read_bytes()).digest()
                    for flow_name, flow in AIG_FLOWS.items():
                        key = (src_digest, flow_name)
                        if key in tried_aig:
                            continue
                        tried_aig.add(key)
                        candidate = run_abc_aig_candidate(
                            abc=abc,
                            source=source,
                            tmp_root=tmp_root,
                            case_name=f"{truth.stem}_{round_idx}_{src_idx}",
                            flow_name=flow_name,
                            flow=flow,
                            timeout=timeout,
                        )
                        if add_candidate(candidate):
                            added_this_round += 1
                if added_this_round == 0:
                    break

    if best is None:
        raise RuntimeError(f"no valid candidate produced for {truth.name}")

    output.write_bytes(best.read_bytes())
    return best


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate optimized AIG files from truth-table benchmarks."
    )
    parser.add_argument(
        "--abc",
        type=Path,
        default=Path(__file__).resolve().with_name("abc"),
        help="Path to the ABC executable.",
    )
    parser.add_argument(
        "--mockturtle",
        type=Path,
        default=repo_root / "mockturtle" / "build" / "examples" / "mockturtle_opt",
        help="Path to the mockturtle_opt executable.",
    )
    parser.add_argument(
        "--benchmarks",
        type=Path,
        default=repo_root / "benchmarks",
        help="Directory containing exNNN.truth files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "output",
        help="Directory where exNNN.aig files will be written.",
    )
    parser.add_argument("--case", help="Optional single case name, for example ex200.")
    parser.add_argument(
        "--effort",
        choices=("quick", "medium", "high"),
        default="medium",
        help="Search effort for BDD orders and ANF phases.",
    )
    parser.add_argument(
        "--orders",
        help="Comma-separated BDD order names. Valid names: natural, reverse, "
        "interleave, interleave_rev, byte_msb, byte_lsb, influence_desc, influence_asc.",
    )
    parser.add_argument(
        "--anf-phases",
        help="Comma-separated ANF/FPRM phase names. Valid names: none, all, lower, upper, alt0, alt1.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each ABC or mockturtle flow.",
    )
    parser.add_argument(
        "--anf-term-cap",
        type=int,
        default=DEFAULT_ANF_TERM_CAP,
        help="Skip ANF/FPRM candidates when any output exceeds this many monomials.",
    )
    parser.add_argument(
        "--no-abc",
        action="store_true",
        help="Disable ABC candidates.",
    )
    parser.add_argument(
        "--no-mockturtle",
        action="store_true",
        help="Disable mockturtle post-optimization candidates.",
    )
    parser.add_argument(
        "--mockturtle-flows",
        default="deep",
        help="Comma-separated mockturtle flows, for example deep,rewrite_balance.",
    )
    parser.add_argument(
        "--mockturtle-top-k",
        type=int,
        default=3,
        help="Run mockturtle on the best K pre-mockturtle candidates.",
    )
    parser.add_argument(
        "--mockturtle-rounds",
        type=int,
        default=1,
        help="Repeat mockturtle post-optimization for this many improvement rounds.",
    )
    parser.add_argument(
        "--ignore-existing",
        action="store_true",
        help="Do not keep the current output as a fallback candidate.",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Simulate existing outputs before using them as candidates.",
    )
    parser.add_argument(
        "--abc-aig-top-k",
        type=int,
        default=3,
        help="Run ABC AIG-refinement flows on the best K candidates after all other synthesis.",
    )
    parser.add_argument(
        "--abc-aig-rounds",
        type=int,
        default=1,
        help="Repeat ABC AIG-refinement for this many improvement rounds (0 to disable).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.benchmarks.is_dir():
        print(f"Benchmark directory not found: {args.benchmarks}", file=sys.stderr)
        return 2

    if args.case:
        truth_files = [args.benchmarks / f"{args.case}.truth"]
    else:
        truth_files = sorted(args.benchmarks.glob("ex*.truth"))

    if not truth_files:
        print("No benchmark truth files found.", file=sys.stderr)
        return 2

    mockturtle_flows = [flow.strip() for flow in args.mockturtle_flows.split(",") if flow.strip()]
    if not mockturtle_flows:
        print("--mockturtle-flows must contain at least one flow name.", file=sys.stderr)
        return 2
    if args.mockturtle_top_k < 1:
        print("--mockturtle-top-k must be at least 1.", file=sys.stderr)
        return 2
    if args.mockturtle_rounds < 1:
        print("--mockturtle-rounds must be at least 1.", file=sys.stderr)
        return 2

    use_abc = False
    if not args.no_abc:
        use_abc = executable_available(args.abc, min(args.timeout, 10), ["-c", "quit"])
        if not use_abc:
            print(f"[WARN] ABC is unavailable, skipping ABC candidates: {args.abc}")
    else:
        print("[INFO] ABC disabled by --no-abc")

    use_mockturtle = False
    if not args.no_mockturtle:
        use_mockturtle = executable_available(args.mockturtle, min(args.timeout, 10))
        if not use_mockturtle and args.mockturtle.is_file():
            print(f"[WARN] mockturtle is unavailable, skipping mockturtle candidates: {args.mockturtle}")
    else:
        print("[INFO] mockturtle disabled by --no-mockturtle")

    total_adp = 0
    improved = 0

    for truth in truth_files:
        if not truth.is_file():
            print(f"Missing benchmark: {truth}", file=sys.stderr)
            return 2

        try:
            problem = read_truth_problem(truth)
            order_items = choose_orders(problem, args.effort, args.orders)
            phase_items = make_phase_library(problem, args.effort, args.anf_phases)
        except ValueError as exc:
            print(f"{truth.name}: {exc}", file=sys.stderr)
            return 2

        output = args.output / f"{truth.stem}.aig"
        old_stats = None
        if output.is_file():
            try:
                parsed_inputs, _outputs, _ands = parse_binary_aig(output.read_bytes())
                if parsed_inputs == problem.inputs:
                    old_stats = aig_stats(output.read_bytes())
            except (ValueError, IndexError):
                old_stats = None

        candidate = optimize_case(
            truth=truth,
            output=output,
            abc=args.abc,
            mockturtle=args.mockturtle,
            order_items=order_items,
            phase_items=phase_items,
            timeout=args.timeout,
            use_abc=use_abc,
            use_mockturtle=use_mockturtle,
            mockturtle_flows=mockturtle_flows,
            mockturtle_top_k=args.mockturtle_top_k,
            mockturtle_rounds=args.mockturtle_rounds,
            keep_existing=not args.ignore_existing,
            verify_existing=args.verify_existing,
            anf_term_cap=args.anf_term_cap,
            abc_aig_top_k=args.abc_aig_top_k,
            abc_aig_rounds=args.abc_aig_rounds,
        )

        if old_stats is not None and candidate.stats.adp < old_stats.adp:
            improved += 1
        total_adp += candidate.stats.adp
        print(
            f"[BEST] {truth.stem}: {candidate.name:<18} "
            f"inputs={problem.inputs:<2} "
            f"area={candidate.stats.area:<7} delay={candidate.stats.delay:<3} "
            f"adp={candidate.stats.adp}"
        )

    print(f"Generated {len(truth_files)} AIG file(s) in {args.output}")
    print(f"Improved cases this run: {improved}")
    print(f"Total local ADP estimate: {total_adp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())