# Latency Benchmarking

## Goal

Build a reusable timing utility for OmniServe inference workloads.

## What I Built

I created `src/runtime/latency_timer.py`, which can benchmark any zero-argument callable.

It reports:

- average latency
- minimum latency
- maximum latency
- p50 latency
- p95 latency
- p99 latency
- throughput per second
- raw latency samples

I also created `benchmarks/tiny_classifier_latency.py`, which benchmarks a tiny classifier across multiple batch sizes.

## Why This Matters

OmniServe is an inference runtime. Inference systems must measure latency and throughput, not just correctness.

A model that works but responds too slowly may not be useful in a production setting.

## Key Concepts

### Latency

Latency is the time required to complete one operation.

### Throughput

Throughput is the number of operations completed per second.

### Warmup

Warmup runs are excluded from measurements because early runs may include initialization overhead.

### p50, p95, p99

Percentile metrics describe the latency distribution.

- p50: median-like typical latency
- p95: slow request boundary for 95% of requests
- p99: tail latency

### GPU Synchronization

CUDA operations are asynchronous, so benchmark code must synchronize before stopping the timer.

## Benchmark Experiment

The benchmark compares batch sizes:

```text
1, 2, 4, 8, 16, 32