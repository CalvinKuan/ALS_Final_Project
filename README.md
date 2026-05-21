# 2026 Advanced Logic Synthesis - Final Project: AI-Based AIG Optimization

**Instructor**: Prof. TingTing Hwang

**Teaching Assistants**: Wuqian Tang, Ruijie Wang

**Contact**: wqtang@cs.nthu.edu.tw, wrj651121@gmail.com

**Tentative Due Date: 2026/06/15 (Mon.) 23:59:59**

## Background

Logic synthesis is a key stage in the digital IC design flow. Given a Boolean function or an RTL description, a synthesis tool transforms the design into an optimized gate-level representation while preserving functional correctness.

In this project, each benchmark is given as a truth table, and your task is to synthesize an equivalent **And-Inverter Graph (AIG)**. An AIG represents Boolean logic using two-input AND nodes and complemented edges. This compact representation is widely used in logic optimization, equivalence checking, technology mapping, and formal verification.

The quality of an AIG is measured by two main objectives:

1. **Area**: the number of internal two-input AND nodes.
2. **Delay**: the number of AIG logic levels.

This project uses **area-delay product (ADP)** as the optimization target:

```text
ADP = area * delay
```

You are required to use an AI-based or LLM-based method to help develop, guide, search, or improve your synthesis flow.

## Overview

You are given 100 Boolean-function benchmarks in `benchmarks/`, named `ex200.truth` through `ex299.truth`. For each input truth table, generate one functionally equivalent AIG file under `output/`.

This final project has two parts:

1. Generate optimized equivalent AIGs for all benchmarks.
2. Write a `report.pdf` explaining how AI/LLM techniques were used and how to reproduce your results.

Your submitted source code will be manually reviewed. All code used to produce your AIG files must be placed inside `student/`.

## Getting Started

Extract the assignment package and enter the working directory:

```bash
tar -xzf ALS_Final_Project.tar.gz
cd ALS_Final_Project
```

The package layout:

```text
ALS_Final_Project/
|-- README.md              # This file
|-- benchmarks/            # Input truth tables, ex200.truth ... ex299.truth
|-- output/                # Your generated AIG files, ex200.aig ... ex299.aig
|-- student/               # Your source code and tools
|   |-- abc                # Provided ABC executable
|   `-- optimizer.py       # Baseline example optimizer
|-- evaluate.py            # Local equivalence and ADP checker
`-- compress.sh            # Submission packaging script
```

Run the provided baseline optimizer:

```bash
python3 student/optimizer.py
```

This generates one baseline AIG for each benchmark in `output/`.

To generate a single case:

```bash
python3 student/optimizer.py --case ex200
```

You are expected to improve this baseline with your own AI/LLM-based optimization method.

## Input Format

Each benchmark is a truth-table file:

```text
benchmarks/exNNN.truth
```

where `NNN` ranges from `200` to `299`.

The truth tables should be read by ABC using:

```text
read_truth -xf benchmarks/exNNN.truth
```

## Output Format

For each input benchmark, submit exactly one AIG file:

```text
output/exNNN.aig
```

Rules:

- The output file name must match the benchmark name.
- The required file-name format is exactly `exNNN.aig`.
- For example, the output for `benchmarks/ex200.truth` must be `output/ex200.aig`.
- Extra AIG files with other names may be rejected by the evaluation script.

## Correctness

Functional correctness is mandatory. If an AIG is not equivalent to the corresponding truth table, that case receives **0 points**.

The equivalence check uses ABC:

```bash
./student/abc -c "read_truth -xf benchmarks/ex200.truth; st; &get; &cec -t output/ex200.aig"
```

The expected successful result is:

```text
Networks are equivalent.
```

## Student Implementation

Work inside `student/`. You may add any number of source files, scripts, prompts, logs, or helper programs, as long as they are needed to reproduce or explain your method.

The provided `student/optimizer.py` is only a minimal example. It uses the simple ABC flow:

```text
read_truth -xf <benchmark>; st; write_aiger -s <output>
```

You may replace it with your own implementation or extend it. Your method may use ABC commands, custom scripts, search algorithms, reinforcement learning, LLM-generated synthesis strategies, prompt-guided optimization, or other AI-based approaches.

> **WARNING**: TA will manually review your source code and report. If you use any technique that circumvents the intended task, such as hardcoding hidden grading outcomes or bypassing equivalence requirements, your score may be reduced to **0 points**.

## Local Evaluation

Run the local evaluator from the project root:

```bash
python3 evaluate.py
```

The script checks every `output/exNNN.aig` against the corresponding truth table. If the AIG is equivalent, it reports area, delay, and ADP.

To evaluate a single case:

```bash
python3 evaluate.py --case ex200
```

The evaluator uses the same correctness command described above, then measures area and delay with ABC `ps`.

## Report

Write your report as `report.pdf` only. Do not submit `.doc` or `.docx` files.

Your report must include at least:

1. How to compile or set up your program.
2. How to execute your optimizer and reproduce your AIG outputs.
3. The overall architecture of your method.
4. How AI/LLM techniques were used in your workflow.
5. Any prompts, models, scripts, or search strategies used.
6. Experimental results, including area, delay, and ADP observations.
7. Difficulties, limitations, and possible improvements.

Upload `report.pdf` separately to eeclass. Do not include it inside the `.tar.gz` archive.

## Grading

| Component | Weight |
|---|---:|
| AIG quality | 80% |
| AI/LLM methodology report and source-code review | 20% |

### Demo Requirement

You must attend the final project demo session in person. The TA will confirm your score during the demo.

If you do not attend the demo, this final project will receive **0 points**, even if you submitted all required files to eeclass.

The detailed demo schedule will be announced later on eeclass.

### AIG Quality Score

There are 100 cases. Each case is worth 1 point before scaling.

For each case, only functionally equivalent AIGs are considered. Among all students' equivalent submissions, the smallest ADP for that case is denoted as `ADP_min` and receives 1 point.

For another equivalent submission with ADP `ADP_yours`, the case score is:

```text
ADP_min / ADP_yours
```

The total AIG-quality raw score is the sum over all 100 cases, then scaled to 80% of the final grade.

Incorrect or missing AIGs receive 0 points for that case.

## Submission

### Step 1: Complete your code and outputs

Before packaging, make sure:

- `output/` contains `ex200.aig` through `ex299.aig`.
- Every output file name follows `exNNN.aig`.
- `python3 evaluate.py` reports equivalence for all 100 cases.
- All source code used to generate the AIGs is under `student/`.
- `report.pdf` explains how to compile and run your method.

### Step 2: Package your submission

Run from the project root:

```bash
./compress.sh <Your_Student_ID>
# Example:
./compress.sh 112062873
```

This creates:

```text
<Student_ID>.tar.gz
```

The archive contains:

```text
output/
student/
```

### Step 3: Submit to eeclass

Upload the following **two files** to eeclass:

1. **`<Student_ID>.tar.gz`** - generated by `compress.sh`.
2. **`report.pdf`** - uploaded separately, not inside the tar.gz.

You must also attend the demo session. Missing the demo means this assignment receives **0 points**.

### Late Penalty

- Within 24 hours: **20% off**
- Within 48 hours: **40% off**
- More than 2 days: **0 points**

## Contact

- **TA**: Wuqian Tang, Ruijie Wang
- **Email**: wqtang@cs.nthu.edu.tw, wrj651121@gmail.com
