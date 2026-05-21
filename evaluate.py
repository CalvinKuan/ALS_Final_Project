#!/usr/bin/env python3
"""Evaluate submitted AIG files for the ALS 2026 final project."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PS_RE = re.compile(r"and\s*=\s*(\d+)\s+lev\s*=\s*(\d+)")


@dataclass
class CaseResult:
    case: str
    status: str
    area: int | None = None
    delay: int | None = None
    adp: int | None = None
    message: str = ""


def run_abc(abc: Path, command: str, timeout: int) -> tuple[int, str]:
    result = subprocess.run(
        [str(abc), "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout


def is_equivalent(abc: Path, truth: Path, aig: Path, timeout: int) -> tuple[bool, str]:
    command = f"read_truth -xf {truth}; st; &get; &cec -t {aig}"
    code, output = run_abc(abc, command, timeout)
    if code != 0:
        return False, output.strip()
    return "Networks are equivalent" in output, output.strip()


def measure_adp(abc: Path, aig: Path, timeout: int) -> tuple[int, int, int]:
    code, output = run_abc(abc, f"read {aig}; ps", timeout)
    if code != 0:
        raise RuntimeError(output.strip())
    match = PS_RE.search(output)
    if not match:
        raise RuntimeError(f"Cannot parse ABC statistics:\n{output}")
    area = int(match.group(1))
    delay = int(match.group(2))
    return area, delay, area * delay


def evaluate_case(abc: Path, truth: Path, output_dir: Path, timeout: int) -> CaseResult:
    case = truth.stem
    aig = output_dir / f"{case}.aig"
    if not aig.is_file():
        return CaseResult(case=case, status="MISSING", message=f"{aig.name} not found")

    try:
        equivalent, message = is_equivalent(abc, truth, aig, timeout)
    except subprocess.TimeoutExpired:
        return CaseResult(case=case, status="TIMEOUT", message="equivalence check timeout")

    if not equivalent:
        return CaseResult(case=case, status="NOT_EQUIV", message=message.splitlines()[-1] if message else "")

    try:
        area, delay, adp = measure_adp(abc, aig, timeout)
    except subprocess.TimeoutExpired:
        return CaseResult(case=case, status="TIMEOUT", message="ADP measurement timeout")
    except RuntimeError as exc:
        return CaseResult(case=case, status="ERROR", message=str(exc))

    return CaseResult(case=case, status="OK", area=area, delay=delay, adp=adp)


def check_output_filenames(output_dir: Path) -> list[str]:
    if not output_dir.is_dir():
        return [f"Output directory not found: {output_dir}"]
    bad = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and not re.fullmatch(r"ex\d{3}\.aig", path.name):
            bad.append(path.name)
    return bad


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Check equivalence and report ADP for output/exNNN.aig files."
    )
    parser.add_argument("--abc", type=Path, default=root / "student" / "abc")
    parser.add_argument("--benchmarks", type=Path, default=root / "benchmarks")
    parser.add_argument("--output", type=Path, default=root / "output")
    parser.add_argument("--case", help="Optional single case name, for example ex200.")
    parser.add_argument("--timeout", type=int, default=60, help="ABC timeout per step in seconds.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.abc.is_file():
        print(f"ABC executable not found: {args.abc}", file=sys.stderr)
        return 2
    if not args.benchmarks.is_dir():
        print(f"Benchmark directory not found: {args.benchmarks}", file=sys.stderr)
        return 2

    bad_names = check_output_filenames(args.output)
    if bad_names:
        print("Invalid output filename(s):")
        for name in bad_names:
            print(f"  {name}")
        print("Expected format: exNNN.aig")
        return 2

    if args.case:
        truth_files = [args.benchmarks / f"{args.case}.truth"]
    else:
        truth_files = sorted(args.benchmarks.glob("ex*.truth"))

    print(f"{'case':<8} {'status':<10} {'area':>10} {'delay':>8} {'adp':>14}")
    print("-" * 54)
    results = []
    for truth in truth_files:
        result = evaluate_case(args.abc, truth, args.output, args.timeout)
        results.append(result)
        area = "-" if result.area is None else str(result.area)
        delay = "-" if result.delay is None else str(result.delay)
        adp = "-" if result.adp is None else str(result.adp)
        print(f"{result.case:<8} {result.status:<10} {area:>10} {delay:>8} {adp:>14}")
        if result.message and result.status != "OK":
            print(f"  {result.message}")

    ok = [r for r in results if r.status == "OK"]
    print("-" * 54)
    print(f"Equivalent cases: {len(ok)}/{len(results)}")
    if ok:
        print(f"Total ADP over equivalent cases: {sum(r.adp or 0 for r in ok)}")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
