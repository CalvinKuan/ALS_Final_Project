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
import csv
import functools
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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


DEFAULT_ORDERS = ("natural", "reverse", "interleave", "interleave_rev", "byte_msb", "influence_desc", "influence_asc")
QUICK_ORDERS = ("reverse", "byte_msb")
DEFAULT_ANF_TERM_CAP = 1024


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


@dataclass(frozen=True)
class ReverseVerilogMatch:
    name: str
    expression: str


@dataclass(frozen=True)
class OptimizerConfig:
    abc: Path
    yosys: Path
    mockturtle: Path
    output: Path
    effort: str
    orders: str | None
    anf_phases: str | None
    timeout: int
    use_abc: bool
    use_yosys: bool
    use_mockturtle: bool
    mockturtle_flows: tuple[str, ...]
    mockturtle_top_k: int
    mockturtle_rounds: int
    portfolio_cycles: int
    keep_existing: bool
    verify_existing: bool
    anf_term_cap: int
    abc_aig_top_k: int
    abc_aig_rounds: int
    max_workers: int | None
    pareto_dir: Path | None
    dump_rev_verilog: Path | None
    cec_final: bool


@dataclass(frozen=True)
class CaseRunResult:
    case: str
    candidate_name: str
    inputs: int
    stats: AigStats
    old_stats: AigStats | None = None

    @property
    def improved(self) -> bool:
        return self.old_stats is not None and self.stats.adp < self.old_stats.adp


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

class ReverseEngineerLibrary:
    """Small structural recognizer for common 16-input arithmetic/bitwise blocks."""

    def __init__(self, problem: TruthProblem) -> None:
        self.problem = problem
        self.builder = AigBuilder(problem.inputs, "optimizer.py reverse-engineered")
        self.truth_mask = (1 << problem.rows) - 1
        self.truth_to_lit: dict[int, int] = {0: 0, self.truth_mask: 1}
        input_truths = input_truth_vectors(problem.inputs)
        self.inputs_by_index: list[tuple[int, int]] = []
        for idx in range(problem.inputs):
            truth = input_truths[2 * (idx + 1)]
            lit = 2 * (idx + 1)
            self.inputs_by_index.append((truth, lit))
            self.truth_to_lit.setdefault(truth, lit)
            self.truth_to_lit.setdefault(self.truth_mask ^ truth, lit ^ 1)

    def _remember(self, truth: int, lit: int) -> int:
        self.truth_to_lit.setdefault(truth, lit)
        return lit

    def inv(self, item: tuple[int, int]) -> tuple[int, int]:
        truth, lit = item
        return self.truth_mask ^ truth, lit ^ 1

    def and2(self, lhs: tuple[int, int], rhs: tuple[int, int]) -> tuple[int, int]:
        truth = lhs[0] & rhs[0]
        lit = self.builder.mk_and(lhs[1], rhs[1])
        return truth, self._remember(truth, lit)

    def or2(self, lhs: tuple[int, int], rhs: tuple[int, int]) -> tuple[int, int]:
        truth = lhs[0] | rhs[0]
        lit = self.builder.mk_or(lhs[1], rhs[1])
        return truth, self._remember(truth, lit)

    def xor2(self, lhs: tuple[int, int], rhs: tuple[int, int]) -> tuple[int, int]:
        truth = lhs[0] ^ rhs[0]
        lit = self.builder.mk_xor(lhs[1], rhs[1])
        return truth, self._remember(truth, lit)

    def input_bit(self, idx: int) -> tuple[int, int]:
        return self.inputs_by_index[idx]

    def const0(self) -> tuple[int, int]:
        return 0, 0

    def const1(self) -> tuple[int, int]:
        return self.truth_mask, 1

    def add_vector_signals(self, lhs: list[tuple[int, int]], rhs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        carry = self.const0()
        outputs: list[tuple[int, int]] = []
        for a_bit, b_bit in zip(lhs, rhs):
            axb = self.xor2(a_bit, b_bit)
            sum_bit = self.xor2(axb, carry)
            carry = self.or2(self.and2(a_bit, b_bit), self.and2(carry, axb))
            self._remember(sum_bit[0], sum_bit[1])
            self.truth_to_lit.setdefault(sum_bit[0], sum_bit[1])
            outputs.append(sum_bit)
        outputs.append(carry)
        return outputs

    def sub_vector_signals(self, lhs: list[tuple[int, int]], rhs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        borrow = self.const0()
        outputs: list[tuple[int, int]] = []
        for a_bit, b_bit in zip(lhs, rhs):
            axb = self.xor2(a_bit, b_bit)
            diff = self.xor2(axb, borrow)
            not_a = self.inv(a_bit)
            not_axb = self.inv(axb)
            borrow = self.or2(self.and2(not_a, b_bit), self.and2(not_axb, borrow))
            outputs.append(diff)
        outputs.append(borrow)
        return outputs

    def add_fixed_width(self, lhs: list[tuple[int, int]], rhs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        carry = self.const0()
        outputs: list[tuple[int, int]] = []
        for a_bit, b_bit in zip(lhs, rhs):
            axb = self.xor2(a_bit, b_bit)
            outputs.append(self.xor2(axb, carry))
            carry = self.or2(self.and2(a_bit, b_bit), self.and2(carry, axb))
        return outputs

    def covered_literals(self, outputs: list[int]) -> list[int] | None:
        literals: list[int] = []
        for truth in outputs:
            lit = self.truth_to_lit.get(truth)
            if lit is None:
                return None
            literals.append(lit)
        return literals

    def byte_inputs(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        return [self.input_bit(i) for i in range(8)], [self.input_bit(i) for i in range(8, 16)]

    def build_bitwise(self) -> None:
        lo, hi = self.byte_inputs()
        for lhs, rhs in ((lo, hi), (hi, lo)):
            for op in (self.and2, self.or2, self.xor2):
                bits = [op(a_bit, b_bit) for a_bit, b_bit in zip(lhs, rhs)]
                for bit in bits:
                    self._remember(bit[0], bit[1])
                    self._remember(self.inv(bit)[0], self.inv(bit)[1])

    def build_add_sub(self) -> None:
        lo, hi = self.byte_inputs()
        zero = self.const0()
        for lhs, rhs in ((lo, hi), (hi, lo)):
            add_bits = self.add_vector_signals(lhs, rhs)
            sub_bits = self.sub_vector_signals(lhs, rhs)
            for bit in add_bits + sub_bits:
                self._remember(bit[0], bit[1])
            for _ in range(16 - len(add_bits)):
                add_bits.append(zero)
            for _ in range(16 - len(sub_bits)):
                sub_bits.append(zero)

    def build_product(self) -> None:
        lo, hi = self.byte_inputs()
        zero = self.const0()
        product = [zero for _ in range(16)]
        for j, b_bit in enumerate(hi):
            partial = [zero for _ in range(16)]
            for i, a_bit in enumerate(lo):
                partial[i + j] = self.and2(a_bit, b_bit)
            product = self.add_fixed_width(product, partial)
        for bit in product:
            self._remember(bit[0], bit[1])

    def build_eq(self) -> None:
        lo, hi = self.byte_inputs()
        eq_bits = [self.inv(self.xor2(a_bit, b_bit)) for a_bit, b_bit in zip(lo, hi)]
        eq = self.builder.mk_balanced_and([lit for _truth, lit in eq_bits])
        eq_truth = self.truth_mask
        for truth, _lit in eq_bits:
            eq_truth &= truth
        self._remember(eq_truth, eq)

    def build_compare(self) -> None:
        lo, hi = self.byte_inputs()
        for lhs, rhs in ((lo, hi), (hi, lo)):
            equal_prefix = self.const1()
            less_terms: list[tuple[int, int]] = []
            greater_terms: list[tuple[int, int]] = []
            for a_bit, b_bit in zip(reversed(lhs), reversed(rhs)):
                less_terms.append(self.and2(equal_prefix, self.and2(self.inv(a_bit), b_bit)))
                greater_terms.append(self.and2(equal_prefix, self.and2(a_bit, self.inv(b_bit))))
                equal_prefix = self.and2(equal_prefix, self.inv(self.xor2(a_bit, b_bit)))
            less = less_terms[0]
            greater = greater_terms[0]
            for term in less_terms[1:]:
                less = self.or2(less, term)
            for term in greater_terms[1:]:
                greater = self.or2(greater, term)
            self._remember(less[0], less[1])
            self._remember(greater[0], greater[1])


def synthesize_reverse_engineered_candidate(problem: TruthProblem) -> Candidate | None:
    """Guess common circuit structure from the truth table and emit a shared AIG."""
    if problem.inputs != 16 or len(problem.outputs) != 16:
        return None
    attempts = (
        (),
        ("bitwise",),
        ("addsub",),
        ("product",),
        ("eq",),
        ("compare",),
    )
    best: Candidate | None = None
    for stages in attempts:
        library = ReverseEngineerLibrary(problem)
        for stage in stages:
            if stage == "bitwise":
                library.build_bitwise()
            elif stage == "addsub":
                library.build_add_sub()
            elif stage == "product":
                library.build_product()
            elif stage == "eq":
                library.build_eq()
            elif stage == "compare":
                library.build_compare()
        outputs = library.covered_literals(problem.outputs)
        if outputs is None:
            continue
        stats = library.builder.stats_for_outputs(outputs)
        data = library.builder.to_binary_aig(outputs)
        candidate = Candidate(name=f"reverse:{'+'.join(stages) or 'wires'}", stats=stats, data=data)
        if better(candidate, best):
            best = candidate
    return best


@functools.lru_cache(maxsize=None)
def reverse_operation_outputs(name: str) -> tuple[int, ...]:
    outputs = [0] * 16
    for row in range(1 << 16):
        a = row & 0xFF
        b = (row >> 8) & 0xFF
        if name == "add":
            value = a + b
        elif name == "sub_ab":
            value = (a - b) & 0x1FF
        elif name == "sub_ba":
            value = (b - a) & 0x1FF
        elif name == "mul":
            value = a * b
        else:
            raise ValueError(f"unknown reverse operation: {name}")
        for bit in range(16):
            if (value >> bit) & 1:
                outputs[bit] |= 1 << row
    return tuple(outputs)


def detect_reverse_verilog_match(problem: TruthProblem) -> ReverseVerilogMatch | None:
    if problem.inputs != 16 or len(problem.outputs) != 16:
        return None
    actual = tuple(problem.outputs)
    operation_exprs = {
        "add": "{1'b0, a} + {1'b0, b}",
        "sub_ab": "{1'b0, a} - {1'b0, b}",
        "sub_ba": "{1'b0, b} - {1'b0, a}",
        "mul": "{8'b0, a} * {8'b0, b}",
    }
    for name, expression in operation_exprs.items():
        if actual == reverse_operation_outputs(name):
            return ReverseVerilogMatch(name=name, expression=expression)
    return None


def reverse_verilog_text(match: ReverseVerilogMatch) -> str:
    inputs = [f"i{i}" for i in range(16)]
    outputs = [f"o{i}" for i in range(16)]
    ports = ", ".join(inputs + outputs)
    input_decl = ", ".join(inputs)
    output_decl = ", ".join(outputs)
    assign_outputs = "\n".join(f"  assign o{i} = y[{i}];" for i in range(16))
    if match.name == "mul":
        body = f"  assign y = {match.expression};"
    else:
        body = (
            f"  wire [8:0] rev_result = {match.expression};\n"
            "  assign y = {7'b0, rev_result};"
        )
    return (
        "// Reverse-engineered Verilog generated by optimizer.py\n"
        f"// Recognized operation: {match.name}\n"
        f"module top({ports});\n"
        f"  input {input_decl};\n"
        f"  output {output_decl};\n"
        "  wire [7:0] a = {i7, i6, i5, i4, i3, i2, i1, i0};\n"
        "  wire [7:0] b = {i15, i14, i13, i12, i11, i10, i9, i8};\n"
        "  wire [15:0] y;\n"
        f"{body}\n"
        f"{assign_outputs}\n"
        "endmodule\n"
    )


def tool_path_arg(path: Path) -> str:
    return '"' + command_path(path).replace('"', '\\"') + '"'


def run_reverse_verilog_candidate(
    yosys: Path,
    abc: Path,
    problem: TruthProblem,
    case_name: str,
    tmp_root: Path,
    dump_dir: Path | None,
    timeout: int,
    run_yosys: bool,
) -> Candidate | None:
    match = detect_reverse_verilog_match(problem)
    if match is None:
        return None

    verilog = reverse_verilog_text(match)
    safe_name = safe_candidate_name(match.name)
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        (dump_dir / f"{case_name}_{safe_name}.v").write_text(verilog, encoding="ascii")

    if not run_yosys:
        return None

    verilog_path = tmp_root / f"{case_name}_{safe_name}_rev.v"
    blif_path = tmp_root / f"{case_name}_{safe_name}_rev.blif"
    aig_path = tmp_root / f"{case_name}_{safe_name}_rev.aig"
    verilog_path.write_text(verilog, encoding="ascii")

    yosys_command = (
        f"read_verilog {tool_path_arg(verilog_path)}; "
        "hierarchy -top top; proc; opt; flatten; opt; techmap; opt; "
        f"write_blif {tool_path_arg(blif_path)}"
    )
    try:
        yosys_result = subprocess.run(
            [str(yosys), "-q", "-p", yosys_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if yosys_result.returncode != 0 or not blif_path.is_file():
        return None

    abc_command = (
        f"read_blif {command_path(blif_path)}; "
        "strash; dc2; "
        f"write_aiger -s {command_path(aig_path)}"
    )
    try:
        abc_result = subprocess.run(
            [str(abc), "-c", abc_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if abc_result.returncode != 0 or not aig_path.is_file():
        return None

    try:
        data = aig_path.read_bytes()
        stats = aig_stats(data)
    except (ValueError, IndexError):
        return None
    return Candidate(name=f"revv:{match.name}", stats=stats, data=data)


def synthesize_byte_lut_candidate(
    problem: TruthProblem,
    ctrl_vars: list[int],
    data_vars: list[int],
    name: str,
    max_unique: int = 160,
    max_terms: int = 1400,
) -> Candidate | None:
    """Decompose f(ctrl,data) into ctrl decoding plus shared 8-bit data LUTs."""
    if problem.inputs != 16 or len(ctrl_vars) != 8 or len(data_vars) != 8:
        return None

    sub_tables_by_output: list[list[int]] = []
    unique_subtables: set[int] = set()
    active_terms = 0
    for truth in problem.outputs:
        output_tables: list[int] = []
        for ctrl in range(256):
            sub_truth = 0
            for data in range(256):
                row = 0
                for bit_idx, var in enumerate(data_vars):
                    if (data >> bit_idx) & 1:
                        row |= 1 << var
                for bit_idx, var in enumerate(ctrl_vars):
                    if (ctrl >> bit_idx) & 1:
                        row |= 1 << var
                if (truth >> row) & 1:
                    sub_truth |= 1 << data
            output_tables.append(sub_truth)
            unique_subtables.add(sub_truth)
            if sub_truth != 0:
                active_terms += 1
            if len(unique_subtables) > max_unique or active_terms > max_terms:
                return None
        sub_tables_by_output.append(output_tables)

    builder = AigBuilder(problem.inputs, f"optimizer.py byte_lut {name}")
    sub_memo: dict[tuple[int, int], int] = {}

    def synth_sub(level: int, remaining: int, truth: int) -> int:
        if truth == 0:
            return 0
        if truth == (1 << (1 << remaining)) - 1:
            return 1
        key = (level, truth)
        cached = sub_memo.get(key)
        if cached is not None:
            return cached
        low_truth, high_truth = split_low_high(truth, remaining)
        condition = 2 * (data_vars[level] + 1)
        lit = builder.mk_ite(
            condition,
            synth_sub(level + 1, remaining - 1, high_truth),
            synth_sub(level + 1, remaining - 1, low_truth),
        )
        sub_memo[key] = lit
        return lit

    ctrl_decode: list[int] = []
    for ctrl in range(256):
        cube = [
            2 * (var + 1) if (ctrl >> bit_idx) & 1 else 2 * (var + 1) + 1
            for bit_idx, var in enumerate(ctrl_vars)
        ]
        ctrl_decode.append(builder.mk_balanced_and(cube))

    def balanced_or(literals: list[int]) -> int:
        if not literals:
            return 0
        layer = literals[:]
        while len(layer) > 1:
            nxt: list[int] = []
            it = iter(layer)
            for lhs in it:
                rhs = next(it, None)
                nxt.append(lhs if rhs is None else builder.mk_or(lhs, rhs))
            layer = nxt
        return layer[0]

    outputs: list[int] = []
    for output_tables in sub_tables_by_output:
        terms: list[int] = []
        for ctrl, sub_truth in enumerate(output_tables):
            sub_lit = synth_sub(0, 8, sub_truth)
            if sub_lit == 0:
                continue
            if sub_lit == 1:
                terms.append(ctrl_decode[ctrl])
            else:
                terms.append(builder.mk_and(ctrl_decode[ctrl], sub_lit))
        outputs.append(balanced_or(terms))

    data = builder.to_binary_aig(outputs)
    return Candidate(name=f"byte_lut:{name}", stats=builder.stats_for_outputs(outputs), data=data)


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
# Cofactor helpers for recursive MUX synthesis
# -----------------------------------------------------------------------------

@functools.lru_cache(maxsize=512)
def _cofactor_mask(var: int, inputs: int) -> int:
    """Bitmask selecting the 2^inputs rows where input variable `var` is 0."""
    step = 1 << var
    rows = 1 << inputs
    m = (1 << step) - 1       # ones in positions [0, step)
    covered = step << 1        # first pattern covers 2*step bits
    while covered < rows:
        m |= m << covered
        covered <<= 1
    return m & ((1 << rows) - 1)


def cofactor0(truth: int, var: int, inputs: int) -> int:
    """f|{xi=0}: duplicate the xi=0 half of each period into the xi=1 half."""
    step = 1 << var
    m = _cofactor_mask(var, inputs)
    bits = truth & m
    return bits | (bits << step)


def cofactor1(truth: int, var: int, inputs: int) -> int:
    """f|{xi=1}: duplicate the xi=1 half of each period into the xi=0 half."""
    step = 1 << var
    rows = 1 << inputs
    m = (~_cofactor_mask(var, inputs)) & ((1 << rows) - 1)
    bits = (truth & m) >> step
    return bits | (bits << step)


# -----------------------------------------------------------------------------
# Sparse SOP/POS candidate
# -----------------------------------------------------------------------------

def synthesize_sparse_candidate(problem: TruthProblem, sparse_cap: int = 8) -> Candidate | None:
    """OR-of-minterms or AND-of-maxterms for very sparse truth tables."""
    builder = AigBuilder(problem.inputs, "optimizer.py sparse")
    outputs: list[int] = []
    rows = problem.rows
    for truth in problem.outputs:
        ones = popcount(truth)
        zeros = rows - ones
        if ones <= sparse_cap:
            and_lits: list[int] = []
            for row in range(rows):
                if not ((truth >> row) & 1):
                    continue
                cube = [2 * (i + 1) if (row >> i) & 1 else 2 * (i + 1) + 1
                        for i in range(problem.inputs)]
                and_lits.append(builder.mk_balanced_and(cube))
            if not and_lits:
                outputs.append(0)
            else:
                lit = and_lits[0]
                for t in and_lits[1:]:
                    lit = builder.mk_or(lit, t)
                outputs.append(lit)
        elif zeros <= sparse_cap:
            or_lits: list[int] = []
            for row in range(rows):
                if (truth >> row) & 1:
                    continue
                clause = [2 * (i + 1) + 1 if (row >> i) & 1 else 2 * (i + 1)
                          for i in range(problem.inputs)]
                cl = clause[0]
                for c in clause[1:]:
                    cl = builder.mk_or(cl, c)
                or_lits.append(cl)
            if not or_lits:
                outputs.append(1)
            else:
                lit = or_lits[0]
                for t in or_lits[1:]:
                    lit = builder.mk_and(lit, t)
                outputs.append(lit)
        else:
            return None
    stats = builder.stats_for_outputs(outputs)
    data = builder.to_binary_aig(outputs)
    return Candidate(name="sparse", stats=stats, data=data)


# -----------------------------------------------------------------------------
# Recursive MUX synthesizer with terminal recognizers
# -----------------------------------------------------------------------------

_MUX_MEMO_LIMIT = 8000


class RecMuxSynth:
    """Shannon-decomposition synthesizer with dynamic variable ordering."""

    def __init__(self, problem: TruthProblem, builder: AigBuilder, sparse_cap: int = 6) -> None:
        self.inputs = problem.inputs
        self.rows = problem.rows
        self.truth_mask = (1 << self.rows) - 1
        self.builder = builder
        self.sparse_cap = sparse_cap
        vals = input_truth_vectors(self.inputs)
        self.input_truths = [vals[2 * (i + 1)] for i in range(self.inputs)]
        # Variable priority: highest influence first
        influences = variable_influences(problem)
        self.var_order = sorted(range(self.inputs), key=lambda b: (-influences[b], b))
        self.memo: dict[int, int] = {0: 0, self.truth_mask: 1}

    def synth(self, truth: int) -> int | None:
        cached = self.memo.get(truth)
        if cached is not None:
            return cached
        if len(self.memo) >= _MUX_MEMO_LIMIT:
            return None
        result = self._synth_impl(truth)
        if result is not None:
            self.memo[truth] = result
        return result

    def _synth_impl(self, truth: int) -> int | None:
        # Literal / negated-literal check
        for idx, it in enumerate(self.input_truths):
            if truth == it:
                return 2 * (idx + 1)
            if truth == (self.truth_mask ^ it):
                return 2 * (idx + 1) + 1

        # Sparse SOP / POS
        ones = popcount(truth)
        if ones <= self.sparse_cap:
            return self._sop(truth)
        if self.rows - ones <= self.sparse_cap:
            return self._pos(truth)

        # Find best variable to split on (skipping don't-cares).
        # _choose_var returns (var, low, high) or None if all vars are don't-cares.
        split = self._choose_var(truth)
        if split is None:
            return None

        var, low, high = split
        low_lit = self.synth(low)
        if low_lit is None:
            return None
        high_lit = self.synth(high)
        if high_lit is None:
            return None
        return self.builder.mk_ite(2 * (var + 1), high_lit, low_lit)

    def _sop(self, truth: int) -> int:
        lits: list[int] = []
        for row in range(self.rows):
            if not ((truth >> row) & 1):
                continue
            cube = [2 * (i + 1) if (row >> i) & 1 else 2 * (i + 1) + 1
                    for i in range(self.inputs)]
            lits.append(self.builder.mk_balanced_and(cube))
        if not lits:
            return 0
        result = lits[0]
        for t in lits[1:]:
            result = self.builder.mk_or(result, t)
        return result

    def _pos(self, truth: int) -> int:
        clauses: list[int] = []
        for row in range(self.rows):
            if (truth >> row) & 1:
                continue
            clause_lits = [2 * (i + 1) + 1 if (row >> i) & 1 else 2 * (i + 1)
                           for i in range(self.inputs)]
            cl = clause_lits[0]
            for c in clause_lits[1:]:
                cl = self.builder.mk_or(cl, c)
            clauses.append(cl)
        if not clauses:
            return 1
        result = clauses[0]
        for t in clauses[1:]:
            result = self.builder.mk_and(result, t)
        return result

    def _choose_var(self, truth: int) -> tuple[int, int, int] | None:
        """Return (var, low, high) for the best split variable, skipping don't-cares.
        Returns None if every variable is a don't-care (shouldn't happen for non-terminal)."""
        best: tuple[int, int, int] | None = None
        best_score = float("inf")
        for var in self.var_order:
            low = cofactor0(truth, var, self.inputs)
            high = cofactor1(truth, var, self.inputs)
            if low == high:
                continue  # don't-care: cofactor unchanged → skipping prevents infinite recursion
            score = 0.0
            for cf in (low, high):
                if cf == 0 or cf == self.truth_mask:
                    score -= 4
                elif cf in self.memo:
                    score -= 2
                elif cf in self.input_truths or (self.truth_mask ^ cf) in self.input_truths:
                    score -= 1
            if best is None or score < best_score:
                best_score = score
                best = (var, low, high)
        return best


def synthesize_mux_candidate(problem: TruthProblem, sparse_cap: int = 6) -> Candidate | None:
    """Recursive MUX synthesis with dynamic variable ordering and terminal recognizers."""
    builder = AigBuilder(problem.inputs, "optimizer.py MUX")
    synth = RecMuxSynth(problem, builder, sparse_cap)
    outputs: list[int] = []
    for truth in problem.outputs:
        lit = synth.synth(truth)
        if lit is None:
            return None
        outputs.append(lit)
    stats = builder.stats_for_outputs(outputs)
    data = builder.to_binary_aig(outputs)
    return Candidate(name="mux", stats=stats, data=data)


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
    # compress2rs: compress2 augmented with resub (well-known ABC best-practice)
    "abc:compress2rs": (
        "st; strash; balance; rewrite; rewrite -z; balance; refactor; refactor -z; "
        "resub; resub -K 6; resub -K 8; balance"
    ),
    # two-pass compress2rs for deeper reduction
    "abc:compress2rs2": (
        "st; strash; balance; rewrite; rewrite -z; balance; refactor; refactor -z; "
        "resub; resub -K 6; resub -K 8; balance; rewrite -z; refactor -z; "
        "resub -K 6 -N 2; resub -K 8 -N 2; balance"
    ),
    # triple &dc2 + dc2 polish
    "abc:aig3": (
        "st; strash; &get; &dc2; &dc2; &dc2; &put; dc2; balance; rewrite -z; refactor -z; balance"
    ),
    # fraig merge then compress2rs
    "abc:fraig_crs": (
        "st; strash; fraig; balance; rewrite; rewrite -z; balance; refactor; refactor -z; "
        "resub; resub -K 6; balance"
    ),
    # level-preserving rewrite (delay-aware, helps ADP)
    "abc:rw_level": (
        "st; strash; balance; rewrite -l; refactor -l; balance; rewrite -l -z; refactor -l -z; balance"
    ),
    # dc2 then level-preserving polish
    "abc:dc2_level": (
        "st; strash; dc2; dc2; balance; rewrite -l -z; refactor -l -z; balance"
    ),
    # long dual-pass with resub
    "abc:long2": (
        "st; strash; balance; rewrite; refactor; rewrite -z; refactor -z; balance; "
        "dc2; balance; rewrite; refactor; rewrite -z; refactor -z; balance; "
        "resub -K 6; resub -K 8; balance; dc2; balance"
    ),
    # don't-care synthesis (merges equivalent nodes using don't-care info)
    "abc:dch": (
        "st; strash; dch; balance; rewrite -z; refactor -z; balance"
    ),
    # dch followed by dc2 for deeper area reduction
    "abc:dch_dc2": (
        "st; strash; dch; dc2; dc2; balance; rewrite -z; refactor -z; balance"
    ),
    # resub with K=10 (finds larger substitution windows)
    "abc:rs_k10": (
        "st; strash; resub -K 10; resub -K 10 -N 2; balance; rewrite -z; refactor -z; balance"
    ),
    # resub with K=12 (very aggressive restructuring for large circuits)
    "abc:rs_k12": (
        "st; strash; resub -K 12; resub -K 12 -N 2; balance; rewrite -z; refactor -z; balance"
    ),
    # three-pass compress2rs (deeper iteration)
    "abc:compress2rs3": (
        "st; strash; balance; rewrite; rewrite -z; balance; refactor; refactor -z; "
        "resub; resub -K 6; resub -K 8; balance; "
        "rewrite; rewrite -z; balance; refactor; refactor -z; resub; resub -K 6; resub -K 8; balance; "
        "rewrite -z; refactor -z; resub -K 6; resub -K 8; balance"
    ),
    # four &dc2 passes (more global AIG restructuring)
    "abc:aig4": (
        "st; strash; &get; &dc2; &dc2; &dc2; &dc2; &put; balance; rewrite -z; refactor -z; balance"
    ),
    # triple dc2 + compress2rs finish
    "abc:adp3": (
        "st; strash; dc2; dc2; dc2; balance; rewrite; refactor; "
        "rewrite -z; refactor -z; resub -K 6; resub -K 8; balance"
    ),
    # three full sweeps with dc2 and resub (best for large circuits)
    "abc:long3": (
        "st; strash; balance; rewrite; refactor; rewrite -z; refactor -z; balance; "
        "dc2; balance; rewrite; refactor; rewrite -z; refactor -z; balance; "
        "dc2; balance; rewrite; refactor; rewrite -z; refactor -z; "
        "resub -K 6; resub -K 8; resub -K 10; balance; dc2; balance"
    ),
    # dch + compress2rs combination
    "abc:dch_crs": (
        "st; strash; dch; balance; "
        "rewrite; rewrite -z; balance; refactor; refactor -z; "
        "resub; resub -K 6; resub -K 8; balance"
    ),
    # logic correspondence (SAT-based equiv merging, very effective for large multi-output)
    "abc:lcorr": (
        "st; strash; lcorr; balance; rewrite -z; refactor -z; balance"
    ),
    # lcorr + dc2 + compress2rs
    "abc:lcorr_crs": (
        "st; strash; lcorr; dc2; balance; "
        "rewrite; rewrite -z; balance; refactor; refactor -z; "
        "resub; resub -K 6; resub -K 8; balance"
    ),
    # ---- ABC9 synthesis flows (&syn2/&syn3/&syn4 explore different AIG structure space) ----
    # &syn2: XOR-rich AIG rewriting (effective for arithmetic/FP circuits)
    "abc:abc9_syn2": (
        "st; strash; &get; &syn2; &put; balance; rewrite -z; refactor -z; balance"
    ),
    # &syn3: 3-input decomposition synthesis
    "abc:abc9_syn3": (
        "st; strash; &get; &syn3; &put; balance; rewrite -z; refactor -z; balance"
    ),
    # &syn4: 4-variable window synthesis
    "abc:abc9_syn4": (
        "st; strash; &get; &syn4; &put; balance; rewrite -z; refactor -z; balance"
    ),
    # combined syn2 + dc2 + syn3 sweep
    "abc:abc9_combo": (
        "st; strash; &get; &syn2; &dc2; &syn3; &put; balance; rewrite -z; refactor -z; balance"
    ),
    # full ABC9 synthesis pipeline
    "abc:abc9_full": (
        "st; strash; &get; &syn2; &dc2; &syn3; &dc2; &syn4; &put; "
        "balance; rewrite -z; refactor -z; balance"
    ),
    # abc9 + classic compress2rs finish
    "abc:abc9_crs": (
        "st; strash; &get; &syn2; &dc2; &syn3; &put; "
        "balance; rewrite; rewrite -z; balance; refactor; refactor -z; "
        "resub; resub -K 6; resub -K 8; balance"
    ),
    # very aggressive: lcorr + triple dc2 + high-K resub (for large multi-output circuits)
    "abc:deep_area": (
        "st; strash; lcorr; dc2; dc2; dc2; balance; rewrite; refactor; "
        "rewrite -z; refactor -z; resub -K 10; resub -K 12; balance; dc2; balance"
    ),
    # area-biased flows: deliberately use less balancing so ABC may keep a
    # smaller, deeper network when that wins ADP.
    "abc:area_dc2": (
        "st; strash; dc2; dc2; dc2; rewrite -z; refactor -z; dc2"
    ),
    "abc:area_dch": (
        "st; strash; dch; dc2; dch; dc2; rewrite -z; refactor -z; dc2"
    ),
    "abc:area_fraig": (
        "st; strash; fraig; dc2; dch; dc2; rewrite -z; refactor -z; dc2"
    ),
    "abc:area_resub": (
        "st; strash; rewrite -z; refactor -z; resub -K 10; resub -K 12; "
        "resub -K 12 -N 2; dc2; dch; dc2"
    ),
    "abc:area_aig": (
        "st; strash; &get; &dc2; &dc2; &dc2; &put; dc2; dch; dc2"
    ),
    # lcorr + abc9_combo (best of both worlds for large circuits)
    "abc:lcorr_abc9": (
        "st; strash; lcorr; &get; &syn2; &dc2; &syn3; &put; "
        "balance; rewrite -z; refactor -z; resub -K 8; balance"
    ),
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
    # compress2rs starting from existing AIG
    "aig:compress2rs": (
        "strash; balance; rewrite; rewrite -z; balance; refactor; refactor -z; "
        "resub; resub -K 6; resub -K 8; balance"
    ),
    # fraig + dc2
    "aig:fraig_dc2": "strash; fraig; dc2; balance; rewrite -z; refactor -z; balance",
    # two full sweeps
    "aig:long2": (
        "strash; balance; rewrite; refactor; rewrite -z; refactor -z; balance; "
        "dc2; balance; rewrite; refactor; rewrite -z; refactor -z; balance"
    ),
    # four dc2 passes
    "aig:dc4": "strash; dc2; dc2; dc2; dc2; balance; rewrite -z; refactor -z; balance",
    # triple &dc2 then dc2 polish
    "aig:aig3_dc": (
        "strash; &get; &dc2; &dc2; &dc2; &put; dc2; balance; rewrite -z; refactor -z; balance"
    ),
    # compress2rs then dc2x
    "aig:crs_dc2": (
        "strash; balance; rewrite; rewrite -z; balance; refactor; refactor -z; "
        "resub; resub -K 6; resub -K 8; dc2; dc2; balance; rewrite -z; refactor -z; balance"
    ),
    # don't-care synthesis on existing AIG
    "aig:dch": "strash; dch; balance; rewrite -z; refactor -z; balance",
    # dch + dc2 on existing AIG (deeper area reduction)
    "aig:dch_dc2": "strash; dch; dc2; dc2; balance; rewrite -z; refactor -z; balance",
    # high-K resub on existing AIG (effective for large circuits)
    "aig:rs_k10": (
        "strash; resub -K 10; resub -K 10 -N 2; balance; rewrite -z; refactor -z; balance"
    ),
    # three-pass compress2rs on existing AIG
    "aig:compress2rs3": (
        "strash; balance; rewrite; rewrite -z; balance; refactor; refactor -z; "
        "resub; resub -K 6; resub -K 8; balance; "
        "rewrite; rewrite -z; balance; refactor; refactor -z; resub; resub -K 6; resub -K 8; balance; "
        "rewrite -z; refactor -z; resub -K 6; balance"
    ),
    # four &dc2 passes then dc2 finish on existing AIG
    "aig:aig4_dc2": (
        "strash; &get; &dc2; &dc2; &dc2; &dc2; &put; dc2; balance; rewrite -z; refactor -z; balance"
    ),
    # dch + compress2rs on existing AIG
    "aig:dch_crs": (
        "strash; dch; balance; rewrite; rewrite -z; balance; refactor; refactor -z; "
        "resub; resub -K 6; resub -K 8; balance"
    ),
    # three full sweeps on existing AIG (deep optimization for large circuits)
    "aig:long3": (
        "strash; balance; rewrite; refactor; rewrite -z; refactor -z; balance; "
        "dc2; balance; rewrite; refactor; rewrite -z; refactor -z; balance; "
        "resub -K 6; resub -K 8; resub -K 10; balance; dc2; balance"
    ),
    # logic correspondence on existing AIG (SAT-based, very effective for large circuits)
    "aig:lcorr": "strash; lcorr; balance; rewrite -z; refactor -z; balance",
    # lcorr + dc2 on existing AIG
    "aig:lcorr_dc2": (
        "strash; lcorr; dc2; balance; rewrite -z; refactor -z; balance"
    ),
    # ---- ABC9 AIG-refinement flows ----
    "aig:abc9_syn2": (
        "strash; &get; &syn2; &put; balance; rewrite -z; refactor -z; balance"
    ),
    "aig:abc9_syn3": (
        "strash; &get; &syn3; &put; balance; rewrite -z; refactor -z; balance"
    ),
    "aig:abc9_syn4": (
        "strash; &get; &syn4; &put; balance; rewrite -z; refactor -z; balance"
    ),
    "aig:abc9_combo": (
        "strash; &get; &syn2; &dc2; &syn3; &put; balance; rewrite -z; refactor -z; balance"
    ),
    "aig:abc9_full": (
        "strash; &get; &syn2; &dc2; &syn3; &dc2; &syn4; &put; "
        "balance; rewrite -z; refactor -z; balance"
    ),
    "aig:abc9_crs": (
        "strash; &get; &syn2; &dc2; &syn3; &put; "
        "balance; rewrite; rewrite -z; balance; refactor; refactor -z; "
        "resub; resub -K 6; resub -K 8; balance"
    ),
    # very aggressive area flow for large AIGs
    "aig:deep_area": (
        "strash; lcorr; dc2; dc2; dc2; balance; rewrite; refactor; "
        "rewrite -z; refactor -z; resub -K 10; resub -K 12; balance; dc2; balance"
    ),
    # area-biased refinements.  These intentionally avoid a final balance pass.
    "aig:area_dc2": "strash; dc2; dc2; dc2; rewrite -z; refactor -z; dc2",
    "aig:area_dch": "strash; dch; dc2; dch; dc2; rewrite -z; refactor -z; dc2",
    "aig:area_fraig": "strash; fraig; dc2; dch; dc2; rewrite -z; refactor -z; dc2",
    "aig:area_resub": (
        "strash; rewrite -z; refactor -z; resub -K 10; resub -K 12; "
        "resub -K 12 -N 2; dc2; dch; dc2"
    ),
    "aig:area_aig": "strash; &get; &dc2; &dc2; &dc2; &put; dc2; dch; dc2",
    # lcorr + abc9 on existing AIG
    "aig:lcorr_abc9": (
        "strash; lcorr; &get; &syn2; &dc2; &syn3; &put; "
        "balance; rewrite -z; refactor -z; resub -K 8; balance"
    ),
}

# Effort-based subsets — quick runs far fewer flows but still gets diverse results.
_ABC_QUICK = [
    "abc:rw", "abc:dc2", "abc:rs", "abc:aig",
    "abc:compress2rs", "abc:dch",
    "abc:area_dc2", "abc:area_dch", "abc:area_aig",
]
_ABC_MEDIUM = [
    "abc:baseline", "abc:rw", "abc:rwz", "abc:dc2", "abc:dc2x",
    "abc:rs", "abc:rs2", "abc:aig", "abc:aig2", "abc:compress2",
    "abc:compress2rs", "abc:compress2rs2", "abc:resyn2", "abc:fraig",
    "abc:dch", "abc:dch_dc2", "abc:rs_k10", "abc:lcorr",
    "abc:abc9_syn2", "abc:abc9_syn3", "abc:abc9_combo", "abc:abc9_full",
    "abc:deep_area", "abc:lcorr_abc9",
    "abc:area_dc2", "abc:area_dch", "abc:area_fraig", "abc:area_resub", "abc:area_aig",
]
# _ABC_HIGH uses all of ABC_FLOWS

_AIG_QUICK = [
    "aig:rw", "aig:dc2", "aig:compress2rs", "aig:dch",
    "aig:abc9_syn2",
    "aig:area_dc2", "aig:area_dch", "aig:area_aig",
]
_AIG_MEDIUM = [
    "aig:rw", "aig:dc2", "aig:resyn2", "aig:rs", "aig:dc2x",
    "aig:compress2rs", "aig:fraig_dc2", "aig:dch", "aig:dch_dc2",
    "aig:rs_k10", "aig:lcorr",
    "aig:abc9_syn2", "aig:abc9_syn3", "aig:abc9_combo", "aig:abc9_full",
    "aig:deep_area", "aig:lcorr_abc9",
    "aig:area_dc2", "aig:area_dch", "aig:area_fraig", "aig:area_resub", "aig:area_aig",
]
# _AIG_HIGH uses all of AIG_FLOWS


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


def is_equivalent_by_abc(abc: Path, truth: Path, aig: Path, timeout: int) -> tuple[bool, str]:
    command = f"read_truth -xf {command_path(truth)}; st; &get; &cec -t {command_path(aig)}"
    try:
        result = subprocess.run(
            [str(abc), "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return result.returncode == 0 and "Networks are equivalent" in result.stdout, result.stdout


def executable_available(executable: Path, timeout: int, args: list[str] | None = None) -> bool:
    command = str(executable)
    if not executable.is_file():
        resolved = shutil.which(command)
        if resolved is None:
            return False
        command = resolved
    try:
        result = subprocess.run(
            [command] + (args or []),
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
    input_path = tmp_root / f"{case_name}_{safe_flow}_in.aig"
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


def pareto_front(candidates: list[Candidate]) -> list[Candidate]:
    """Return the Pareto-optimal subset minimizing both area and delay."""
    front: list[Candidate] = []
    for cand in candidates:
        dominated = False
        new_front: list[Candidate] = []
        for p in front:
            if p.stats.area <= cand.stats.area and p.stats.delay <= cand.stats.delay:
                dominated = True
                new_front.append(p)
            elif cand.stats.area <= p.stats.area and cand.stats.delay <= p.stats.delay:
                pass  # cand dominates p, drop p
            else:
                new_front.append(p)
        if not dominated:
            new_front.append(cand)
        front = new_front
    return front


def safe_candidate_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def save_pareto_front(case_name: str, candidates: list[Candidate], pareto_dir: Path) -> None:
    case_dir = pareto_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    front = sorted(
        pareto_front(candidates),
        key=lambda candidate: (
            candidate.stats.area,
            candidate.stats.delay,
            candidate.stats.adp,
            candidate.name,
        ),
    )
    manifest = []
    for index, candidate in enumerate(front):
        filename = (
            f"{index:02d}_{safe_candidate_name(candidate.name)}_"
            f"a{candidate.stats.area}_l{candidate.stats.delay}.aig"
        )
        (case_dir / filename).write_bytes(candidate.read_bytes())
        manifest.append(
            {
                "file": filename,
                "name": candidate.name,
                "area": candidate.stats.area,
                "delay": candidate.stats.delay,
                "adp": candidate.stats.adp,
            }
        )
    (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_reference_csv(path: Path) -> dict[str, AigStats]:
    refs: dict[str, AigStats] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case = row.get("case")
            if not case:
                continue
            refs[case] = AigStats(area=int(row["area"]), delay=int(row["delay"]))
    return refs


def reference_gap_rows(
    truth_files: list[Path],
    output_dir: Path,
    refs: dict[str, AigStats],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for truth in truth_files:
        case = truth.stem
        ref = refs.get(case)
        if ref is None:
            continue
        current: AigStats | None = None
        aig_path = output_dir / f"{case}.aig"
        if aig_path.is_file():
            try:
                current = aig_stats(aig_path.read_bytes())
            except (ValueError, IndexError):
                current = None
        gap = None if current is None else current.adp / ref.adp
        rows.append(
            {
                "case": case,
                "current": current,
                "reference": ref,
                "gap": gap,
                "delta": None if current is None else current.adp - ref.adp,
            }
        )
    return rows


def print_reference_gap_report(rows: list[dict[str, object]], top: int) -> None:
    if not rows:
        return
    ranked = sorted(
        rows,
        key=lambda row: (
            -1.0 if row["gap"] is None else -float(row["gap"]),
            row["case"],
        ),
    )
    print("[REF] worst current/reference ADP gaps:")
    for row in ranked[:top]:
        case = str(row["case"])
        current = row["current"]
        reference = row["reference"]
        if not isinstance(reference, AigStats):
            continue
        if not isinstance(current, AigStats):
            print(
                f"[REF] {case}: missing/invalid output, "
                f"ref_area={reference.area} ref_delay={reference.delay} ref_adp={reference.adp}"
            )
            continue
        gap = float(row["gap"])
        delta = int(row["delta"])
        sign = "+" if delta >= 0 else ""
        print(
            f"[REF] {case}: gap={gap:.2f}x delta={sign}{delta} "
            f"cur=({current.area},{current.delay},{current.adp}) "
            f"ref=({reference.area},{reference.delay},{reference.adp})"
        )


def optimize_case(
    truth: Path,
    output: Path,
    abc: Path,
    yosys: Path,
    mockturtle: Path,
    order_items: list[tuple[str, list[int]]],
    phase_items: list[tuple[str, int]],
    timeout: int,
    use_abc: bool,
    use_yosys: bool,
    use_mockturtle: bool,
    mockturtle_flows: list[str],
    mockturtle_top_k: int,
    mockturtle_rounds: int,
    portfolio_cycles: int,
    keep_existing: bool,
    verify_existing: bool,
    anf_term_cap: int,
    abc_aig_top_k: int = 3,
    abc_aig_rounds: int = 1,
    max_workers: int | None = None,
    effort: str = "medium",
    pareto_dir: Path | None = None,
    dump_rev_verilog: Path | None = None,
    cec_final: bool = False,
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

    # Fast pure-Python candidates first (no subprocess overhead)
    add_candidate(synthesize_reverse_engineered_candidate(problem))
    add_candidate(synthesize_byte_lut_candidate(problem, list(range(8, 16)), list(range(8)), "hi_ctrl"))
    add_candidate(synthesize_byte_lut_candidate(problem, list(range(8)), list(range(8, 16)), "lo_ctrl"))
    with tempfile.TemporaryDirectory(prefix=f"{truth.stem}_revv_", dir=output.parent) as tmp_dir:
        add_candidate(
            run_reverse_verilog_candidate(
                yosys=yosys,
                abc=abc,
                problem=problem,
                case_name=truth.stem,
                tmp_root=Path(tmp_dir),
                dump_dir=dump_rev_verilog,
                timeout=timeout,
                run_yosys=use_yosys and use_abc,
            )
        )
    add_candidate(synthesize_sparse_candidate(problem))
    add_candidate(synthesize_mux_candidate(problem))

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

            def _run_abc_truth_flow(item: tuple[str, str]) -> Candidate | None:
                flow_name, flow = item
                tmp_output = tmp_root / f"{truth.stem}_{flow_name.replace(':', '_')}.aig"
                if not run_abc_flow(abc, truth, tmp_output, flow, timeout):
                    return None
                try:
                    data = tmp_output.read_bytes()
                    parsed_inputs, _outputs, _ands = parse_binary_aig(data)
                    if parsed_inputs != problem.inputs:
                        return None
                    return Candidate(name=flow_name, stats=aig_stats(data), data=data)
                except (ValueError, IndexError):
                    return None

            if effort == "quick":
                selected_abc = {k: ABC_FLOWS[k] for k in _ABC_QUICK if k in ABC_FLOWS}
            elif effort == "high":
                selected_abc = ABC_FLOWS
            else:
                selected_abc = {k: ABC_FLOWS[k] for k in _ABC_MEDIUM if k in ABC_FLOWS}
            workers = min(len(selected_abc), max_workers or os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for cand in pool.map(_run_abc_truth_flow, selected_abc.items()):
                    add_candidate(cand)

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

                mt_tasks: list[tuple[int, Candidate, str]] = []
                for source_index, source in enumerate(sources):
                    source_digest = hashlib.sha256(source.read_bytes()).digest()
                    for flow in mockturtle_flows:
                        key = (source_digest, flow)
                        if key not in tried_mockturtle:
                            tried_mockturtle.add(key)
                            mt_tasks.append((source_index, source, flow))

                if not mt_tasks:
                    break

                def _run_mt_task(task: tuple[int, Candidate, str]) -> Candidate | None:
                    source_index, source, flow = task
                    return run_mockturtle_flow(
                        mockturtle=mockturtle,
                        source=source,
                        tmp_root=tmp_root,
                        case_name=f"{truth.stem}_{round_idx}_{source_index}",
                        flow=flow,
                        timeout=timeout,
                    )

                workers = min(len(mt_tasks), max_workers or os.cpu_count() or 4)
                added_this_round = 0
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for cand in pool.map(_run_mt_task, mt_tasks):
                        if add_candidate(cand):
                            added_this_round += 1

                if added_this_round == 0:
                    break

    # ABC AIG refinement: run AIG-to-AIG ABC flows on the top-k candidates found so
    # far (including mockturtle results).  Different starting points lead ABC to
    # different local optima, often improving on the truth-table starting point.
    # We use both ADP-ranked top-k AND the Pareto front (area vs delay) to maximise
    # the diversity of starting points explored.
    if use_abc and candidates and abc_aig_rounds > 0:
        tried_aig: set[tuple[bytes, str]] = set()
        with tempfile.TemporaryDirectory(prefix=f"{truth.stem}_abc_aig_", dir=output.parent) as tmp_dir:
            tmp_root = Path(tmp_dir)
            for round_idx in range(abc_aig_rounds):
                adp_sorted = sorted(
                    candidates,
                    key=lambda c: (c.stats.adp, c.stats.area, c.stats.delay, c.name),
                )[:abc_aig_top_k]
                area_sorted = sorted(
                    candidates,
                    key=lambda c: (c.stats.area, c.stats.delay, c.stats.adp, c.name),
                )[:abc_aig_top_k]
                delay_sorted = sorted(
                    candidates,
                    key=lambda c: (c.stats.delay, c.stats.area, c.stats.adp, c.name),
                )[:max(1, abc_aig_top_k // 2)]
                pfront = pareto_front(candidates)
                seen_ids: set[int] = set()
                sources: list[Candidate] = []
                for c in adp_sorted + area_sorted + delay_sorted + pfront:
                    if id(c) not in seen_ids:
                        seen_ids.add(id(c))
                        sources.append(c)
                sources = sources[: abc_aig_top_k * 3]

                if effort == "quick":
                    selected_aig = {k: AIG_FLOWS[k] for k in _AIG_QUICK if k in AIG_FLOWS}
                elif effort == "high":
                    selected_aig = AIG_FLOWS
                else:
                    selected_aig = {k: AIG_FLOWS[k] for k in _AIG_MEDIUM if k in AIG_FLOWS}

                aig_tasks: list[tuple[int, Candidate, str, str]] = []
                for src_idx, source in enumerate(sources):
                    src_digest = hashlib.sha256(source.read_bytes()).digest()
                    for flow_name, flow in selected_aig.items():
                        key = (src_digest, flow_name)
                        if key not in tried_aig:
                            tried_aig.add(key)
                            aig_tasks.append((src_idx, source, flow_name, flow))

                if not aig_tasks:
                    break

                def _run_aig_task(task: tuple[int, Candidate, str, str]) -> Candidate | None:
                    src_idx, source, flow_name, flow = task
                    return run_abc_aig_candidate(
                        abc=abc,
                        source=source,
                        tmp_root=tmp_root,
                        case_name=f"{truth.stem}_{round_idx}_{src_idx}",
                        flow_name=flow_name,
                        flow=flow,
                        timeout=timeout,
                    )

                workers = min(len(aig_tasks), max_workers or os.cpu_count() or 4)
                added_this_round = 0
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for cand in pool.map(_run_aig_task, aig_tasks):
                        if add_candidate(cand):
                            added_this_round += 1

                if added_this_round == 0:
                    break

    for cycle_idx in range(max(0, portfolio_cycles - 1)):
        if use_mockturtle and candidates:
            tried_mockturtle: set[tuple[bytes, str]] = set()
            with tempfile.TemporaryDirectory(prefix=f"{truth.stem}_xmt{cycle_idx}_", dir=output.parent) as tmp_dir:
                tmp_root = Path(tmp_dir)
                for round_idx in range(mockturtle_rounds):
                    sources = sorted(
                        pareto_front(candidates),
                        key=lambda candidate: (
                            candidate.stats.adp,
                            candidate.stats.area,
                            candidate.stats.delay,
                            candidate.name,
                        ),
                    )[:mockturtle_top_k]
                    if len(sources) < mockturtle_top_k:
                        for source in sorted(
                            candidates,
                            key=lambda candidate: (
                                candidate.stats.adp,
                                candidate.stats.area,
                                candidate.stats.delay,
                                candidate.name,
                            ),
                        ):
                            if source not in sources:
                                sources.append(source)
                            if len(sources) >= mockturtle_top_k:
                                break

                    mt_tasks: list[tuple[int, Candidate, str]] = []
                    for source_index, source in enumerate(sources):
                        source_digest = hashlib.sha256(source.read_bytes()).digest()
                        for flow in mockturtle_flows:
                            key = (source_digest, flow)
                            if key not in tried_mockturtle:
                                tried_mockturtle.add(key)
                                mt_tasks.append((source_index, source, flow))

                    if not mt_tasks:
                        break

                    def _run_cross_mt_task(task: tuple[int, Candidate, str]) -> Candidate | None:
                        source_index, source, flow = task
                        return run_mockturtle_flow(
                            mockturtle=mockturtle,
                            source=source,
                            tmp_root=tmp_root,
                            case_name=f"{truth.stem}_x{cycle_idx}_{round_idx}_{source_index}",
                            flow=flow,
                            timeout=timeout,
                        )

                    workers = min(len(mt_tasks), max_workers or os.cpu_count() or 4)
                    added_this_round = 0
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        for cand in pool.map(_run_cross_mt_task, mt_tasks):
                            if add_candidate(cand):
                                added_this_round += 1

                    if added_this_round == 0:
                        break

        if use_abc and candidates and abc_aig_rounds > 0:
            tried_aig: set[tuple[bytes, str]] = set()
            with tempfile.TemporaryDirectory(prefix=f"{truth.stem}_xabc{cycle_idx}_", dir=output.parent) as tmp_dir:
                tmp_root = Path(tmp_dir)
                for round_idx in range(abc_aig_rounds):
                    pfront = pareto_front(candidates)
                    sources = sorted(
                        pfront,
                        key=lambda c: (c.stats.adp, c.stats.area, c.stats.delay, c.name),
                    )[:abc_aig_top_k * 2]
                    area_sources = sorted(
                        candidates,
                        key=lambda c: (c.stats.area, c.stats.delay, c.stats.adp, c.name),
                    )[:abc_aig_top_k]
                    for c in area_sources:
                        if c not in sources:
                            sources.append(c)
                    sources = sources[: abc_aig_top_k * 3]

                    if effort == "quick":
                        selected_aig = {k: AIG_FLOWS[k] for k in _AIG_QUICK if k in AIG_FLOWS}
                    elif effort == "high":
                        selected_aig = AIG_FLOWS
                    else:
                        selected_aig = {k: AIG_FLOWS[k] for k in _AIG_MEDIUM if k in AIG_FLOWS}

                    aig_tasks: list[tuple[int, Candidate, str, str]] = []
                    for src_idx, source in enumerate(sources):
                        src_digest = hashlib.sha256(source.read_bytes()).digest()
                        for flow_name, flow in selected_aig.items():
                            key = (src_digest, flow_name)
                            if key not in tried_aig:
                                tried_aig.add(key)
                                aig_tasks.append((src_idx, source, flow_name, flow))

                    if not aig_tasks:
                        break

                    def _run_cross_aig_task(task: tuple[int, Candidate, str, str]) -> Candidate | None:
                        src_idx, source, flow_name, flow = task
                        return run_abc_aig_candidate(
                            abc=abc,
                            source=source,
                            tmp_root=tmp_root,
                            case_name=f"{truth.stem}_x{cycle_idx}_{round_idx}_{src_idx}",
                            flow_name=flow_name,
                            flow=flow,
                            timeout=timeout,
                        )

                    workers = min(len(aig_tasks), max_workers or os.cpu_count() or 4)
                    added_this_round = 0
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        for cand in pool.map(_run_cross_aig_task, aig_tasks):
                            if add_candidate(cand):
                                added_this_round += 1

                    if added_this_round == 0:
                        break

    if best is None:
        raise RuntimeError(f"no valid candidate produced for {truth.name}")

    if pareto_dir is not None:
        save_pareto_front(truth.stem, candidates, pareto_dir)

    output.write_bytes(best.read_bytes())
    if cec_final:
        if not use_abc:
            raise RuntimeError("--cec-final requested but ABC is unavailable")
        equivalent, message = is_equivalent_by_abc(abc, truth, output, timeout)
        if not equivalent:
            tail = message.strip().splitlines()[-1] if message.strip() else "no ABC output"
            raise RuntimeError(f"ABC CEC failed for {truth.name}: {tail}")
    return best


def run_one_case(truth: Path, config: OptimizerConfig) -> CaseRunResult:
    if not truth.is_file():
        raise FileNotFoundError(f"Missing benchmark: {truth}")

    problem = read_truth_problem(truth)
    order_items = choose_orders(problem, config.effort, config.orders)
    phase_items = make_phase_library(problem, config.effort, config.anf_phases)

    output = config.output / f"{truth.stem}.aig"
    old_stats = None
    if output.is_file():
        try:
            output_data = output.read_bytes()
            parsed_inputs, _outputs, _ands = parse_binary_aig(output_data)
            if parsed_inputs == problem.inputs:
                old_stats = aig_stats(output_data)
        except (ValueError, IndexError):
            old_stats = None

    candidate = optimize_case(
        truth=truth,
        output=output,
        abc=config.abc,
        yosys=config.yosys,
        mockturtle=config.mockturtle,
        order_items=order_items,
        phase_items=phase_items,
        timeout=config.timeout,
        use_abc=config.use_abc,
        use_yosys=config.use_yosys,
        use_mockturtle=config.use_mockturtle,
        mockturtle_flows=list(config.mockturtle_flows),
        mockturtle_top_k=config.mockturtle_top_k,
        mockturtle_rounds=config.mockturtle_rounds,
        portfolio_cycles=config.portfolio_cycles,
        keep_existing=config.keep_existing,
        verify_existing=config.verify_existing,
        anf_term_cap=config.anf_term_cap,
        abc_aig_top_k=config.abc_aig_top_k,
        abc_aig_rounds=config.abc_aig_rounds,
        max_workers=config.max_workers,
        effort=config.effort,
        pareto_dir=config.pareto_dir,
        dump_rev_verilog=config.dump_rev_verilog,
        cec_final=config.cec_final,
    )

    return CaseRunResult(
        case=truth.stem,
        candidate_name=candidate.name,
        inputs=problem.inputs,
        stats=candidate.stats,
        old_stats=old_stats,
    )


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
        "--yosys",
        type=Path,
        default=Path("yosys"),
        help="Path to the Yosys executable, or 'yosys' to use PATH.",
    )
    parser.add_argument(
        "--mockturtle",
        type=Path,
        default=Path(__file__).resolve().with_name("mockturtle_opt"),
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
        "--no-yosys",
        action="store_true",
        help="Disable reverse-engineered Verilog synthesis through Yosys.",
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
        default=5,
        help="Run mockturtle on the best K pre-mockturtle candidates.",
    )
    parser.add_argument(
        "--mockturtle-rounds",
        type=int,
        default=3,
        help="Repeat mockturtle post-optimization for this many improvement rounds.",
    )
    parser.add_argument(
        "--portfolio-cycles",
        type=int,
        default=1,
        help="Repeat ABC/ABC9 and mockturtle cross-optimization cycles. "
             "Use 2-3 for stubborn cases.",
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
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel workers per synthesis phase (default: all CPUs). "
             "Set lower (e.g. 2-4) when running multiple optimizer instances in parallel.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of benchmark cases to optimize in parallel. Use 0 for all CPU cores.",
    )
    parser.add_argument(
        "--pareto-dir",
        type=Path,
        default=None,
        help="Optional directory for saving each case's area-delay Pareto-front candidates.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=None,
        help="Optional reference_result.csv for reporting and targeting losing cases.",
    )
    parser.add_argument(
        "--reference-top",
        type=int,
        default=20,
        help="How many worst reference gaps to print when --reference-csv is set.",
    )
    parser.add_argument(
        "--only-reference-lagging",
        action="store_true",
        help="With --reference-csv, optimize only cases whose current ADP is worse than reference.",
    )
    parser.add_argument(
        "--reference-report-only",
        action="store_true",
        help="With --reference-csv, print the gap report and exit without optimizing.",
    )
    parser.add_argument(
        "--reference-gap-threshold",
        type=float,
        default=1.0,
        help="Current/reference ADP threshold for --only-reference-lagging.",
    )
    parser.add_argument(
        "--dump-rev-verilog",
        nargs="?",
        default=None,
        const="",
        help="Dump recognized reverse-engineered Verilog. "
             "With no path, writes under <output>/rev_verilog.",
    )
    parser.add_argument(
        "--cec-final",
        action="store_true",
        help="Run ABC CEC on each final written AIG before reporting success.",
    )
    parser.add_argument(
        "--abc-aig-top-k",
        type=int,
        default=5,
        help="Run ABC AIG-refinement flows on the best K candidates after all other synthesis.",
    )
    parser.add_argument(
        "--abc-aig-rounds",
        type=int,
        default=2,
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

    reference_stats: dict[str, AigStats] | None = None
    if args.only_reference_lagging and args.reference_csv is None:
        print("--only-reference-lagging requires --reference-csv", file=sys.stderr)
        return 2
    if args.reference_report_only and args.reference_csv is None:
        print("--reference-report-only requires --reference-csv", file=sys.stderr)
        return 2
    if args.reference_csv is not None:
        if not args.reference_csv.is_file():
            print(f"Reference CSV not found: {args.reference_csv}", file=sys.stderr)
            return 2
        if args.reference_top < 1:
            print("--reference-top must be at least 1.", file=sys.stderr)
            return 2
        if args.reference_gap_threshold < 0:
            print("--reference-gap-threshold must be non-negative.", file=sys.stderr)
            return 2
        reference_stats = read_reference_csv(args.reference_csv)
        gap_rows = reference_gap_rows(truth_files, args.output, reference_stats)
        print_reference_gap_report(gap_rows, args.reference_top)
        if args.reference_report_only:
            return 0
        if args.only_reference_lagging:
            row_by_case = {str(row["case"]): row for row in gap_rows}
            filtered: list[Path] = []
            for truth in truth_files:
                row = row_by_case.get(truth.stem)
                if row is None:
                    continue
                gap = row["gap"]
                if gap is None or float(gap) > args.reference_gap_threshold:
                    filtered.append(truth)
            truth_files = filtered
            print(f"[REF] optimizing {len(truth_files)} lagging case(s)")
            if not truth_files:
                print("No lagging cases selected.")
                return 0

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
    if args.portfolio_cycles < 1:
        print("--portfolio-cycles must be at least 1.", file=sys.stderr)
        return 2

    use_abc = False
    if not args.no_abc:
        use_abc = executable_available(args.abc, min(args.timeout, 10), ["-c", "quit"])
        if not use_abc:
            print(f"[WARN] ABC is unavailable, skipping ABC candidates: {args.abc}")
    else:
        print("[INFO] ABC disabled by --no-abc")

    use_yosys = False
    if not args.no_yosys:
        use_yosys = executable_available(args.yosys, min(args.timeout, 10), ["-V"])
        if not use_yosys:
            print(f"[WARN] Yosys is unavailable, skipping reverse-Verilog synthesis: {args.yosys}")
    else:
        print("[INFO] Yosys reverse-Verilog synthesis disabled by --no-yosys")

    use_mockturtle = False
    if not args.no_mockturtle:
        use_mockturtle = executable_available(args.mockturtle, min(args.timeout, 10))
        if not use_mockturtle and args.mockturtle.is_file():
            print(f"[WARN] mockturtle is unavailable, skipping mockturtle candidates: {args.mockturtle}")
    else:
        print("[INFO] mockturtle disabled by --no-mockturtle")

    if args.max_workers is not None and args.max_workers < 1:
        print("--max-workers must be at least 1.", file=sys.stderr)
        return 2
    if args.jobs < 0:
        print("--jobs must be at least 0.", file=sys.stderr)
        return 2

    cpu_count = os.cpu_count() or 1
    jobs = cpu_count if args.jobs == 0 else args.jobs
    jobs = max(1, min(jobs, len(truth_files)))
    per_case_workers = args.max_workers
    if per_case_workers is None:
        per_case_workers = max(1, cpu_count // jobs)

    dump_rev_verilog = None
    if args.dump_rev_verilog is not None:
        dump_rev_verilog = args.output / "rev_verilog" if args.dump_rev_verilog == "" else Path(args.dump_rev_verilog)

    config = OptimizerConfig(
        abc=args.abc,
        yosys=args.yosys,
        mockturtle=args.mockturtle,
        output=args.output,
        effort=args.effort,
        orders=args.orders,
        anf_phases=args.anf_phases,
        timeout=args.timeout,
        use_abc=use_abc,
        use_yosys=use_yosys,
        use_mockturtle=use_mockturtle,
        mockturtle_flows=tuple(mockturtle_flows),
        mockturtle_top_k=args.mockturtle_top_k,
        mockturtle_rounds=args.mockturtle_rounds,
        portfolio_cycles=args.portfolio_cycles,
        keep_existing=not args.ignore_existing,
        verify_existing=args.verify_existing,
        anf_term_cap=args.anf_term_cap,
        abc_aig_top_k=args.abc_aig_top_k,
        abc_aig_rounds=args.abc_aig_rounds,
        max_workers=per_case_workers,
        pareto_dir=args.pareto_dir,
        dump_rev_verilog=dump_rev_verilog,
        cec_final=args.cec_final,
    )

    print(
        f"[INFO] cases={len(truth_files)} jobs={jobs} "
        f"workers_per_case={per_case_workers} effort={args.effort}"
    )

    results: list[CaseRunResult] = []

    def print_result(result: CaseRunResult) -> None:
        print(
            f"[BEST] {result.case}: {result.candidate_name:<18} "
            f"inputs={result.inputs:<2} "
            f"area={result.stats.area:<7} delay={result.stats.delay:<3} "
            f"adp={result.stats.adp}"
        )

    if jobs == 1:
        for truth in truth_files:
            try:
                result = run_one_case(truth, config)
            except Exception as exc:
                print(f"[ERROR] {truth.stem}: {exc}", file=sys.stderr)
                return 1
            results.append(result)
            print_result(result)
    else:
        failures = 0
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(run_one_case, truth, config): truth for truth in truth_files}
            for future in as_completed(futures):
                truth = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failures += 1
                    print(f"[ERROR] {truth.stem}: {exc}", file=sys.stderr)
                    continue
                results.append(result)
                print_result(result)
        if failures:
            print(f"Failed cases: {failures}", file=sys.stderr)
            return 1

    total_adp = sum(result.stats.adp for result in results)
    improved = sum(1 for result in results if result.improved)
    print(f"Generated {len(truth_files)} AIG file(s) in {args.output}")
    print(f"Improved cases this run: {improved}")
    print(f"Total local ADP estimate: {total_adp}")
    if reference_stats is not None:
        print_reference_gap_report(reference_gap_rows(truth_files, args.output, reference_stats), args.reference_top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
