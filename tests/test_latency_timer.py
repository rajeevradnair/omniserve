from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pytest

from src.runtime.latency_timer import percentile, benchmark_callable


def test_percentile_returns_expected_values():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert percentile(values, 50) == 3.0
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 5.0


def test_benchmark_callable_returns_summary_fields():
    def tiny_function():
        return 1 + 1

    result = benchmark_callable(
        fn=tiny_function,
        warmup=1,
        runs=3,
        device="cpu",
    )

    assert result["warmup_runs"] == 1
    assert result["measured_runs"] == 3
    assert "avg_ms" in result
    assert "p50_ms" in result
    assert "p95_ms" in result
    assert "p99_ms" in result
    assert "throughput_per_second" in result
    assert len(result["raw_latencies_ms"]) == 3


def test_benchmark_callable_rejects_invalid_runs():
    def tiny_function():
        return 1 + 1

    with pytest.raises(ValueError):
        benchmark_callable(fn=tiny_function, warmup=0, runs=0)


def test_benchmark_callable_rejects_negative_warmup():
    def tiny_function():
        return 1 + 1

    with pytest.raises(ValueError):
        benchmark_callable(fn=tiny_function, warmup=-1, runs=1)