#!/usr/bin/env python3
"""Quick test of the new synthesizers on a few cases."""
import sys, time
sys.path.insert(0, 'student')
from optimizer import (
    read_truth_problem, synthesize_mux_candidate, synthesize_sparse_candidate,
    synthesize_bdd_candidate, make_order_library, is_equivalent_by_simulation, aig_stats
)
from pathlib import Path

test_cases = ['ex265', 'ex275', 'ex266', 'ex261', 'ex270', 'ex255', 'ex260', 'ex271']

for name in test_cases:
    truth_path = Path(f'benchmarks/{name}.truth')
    if not truth_path.exists():
        continue
    problem = read_truth_problem(truth_path)

    # Current best
    aig_path = Path(f'output/{name}.aig')
    cur_adp = None
    if aig_path.exists():
        d = aig_path.read_bytes()
        s = aig_stats(d)
        cur_adp = s.adp

    # Sparse
    t0 = time.time()
    sc = synthesize_sparse_candidate(problem)
    t_sparse = time.time() - t0

    # MUX
    t0 = time.time()
    mc = synthesize_mux_candidate(problem)
    t_mux = time.time() - t0

    print(f'{name} (inputs={problem.inputs}, outputs={len(problem.outputs)}):')
    print(f'  current_adp={cur_adp}')
    if sc:
        ok = is_equivalent_by_simulation(sc.read_bytes(), problem)
        print(f'  sparse:  area={sc.stats.area}, delay={sc.stats.delay}, adp={sc.stats.adp}, correct={ok}, t={t_sparse:.3f}s')
    else:
        print(f'  sparse:  N/A (not sparse enough)')
    if mc:
        ok = is_equivalent_by_simulation(mc.read_bytes(), problem)
        print(f'  mux:     area={mc.stats.area}, delay={mc.stats.delay}, adp={mc.stats.adp}, correct={ok}, t={t_mux:.3f}s')
    else:
        print(f'  mux:     N/A (memo limit hit)')
    print()
