import time
from typing import Callable, Any

import torch


def percentile(values: list[float], percent: float) -> float:
    """
    Compute a percentile from a list of numeric values.

    Example:
        percent=50 means p50
        percent=95 means p95
        percent=99 means p99
    """

    if not values:
        raise ValueError("Cannot compute percentile of an empty list.")

    sorted_values = sorted(values)

    index = int(round((percent / 100) * (len(sorted_values) - 1)))

    return sorted_values[index]


def synchronize_if_needed(device: str | torch.device | None = None) -> None:
    """
    Synchronize CUDA work if the benchmark is using a CUDA device.

    This prevents under-measuring GPU latency.
    """

    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return

    device_obj = torch.device(device)

    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)


def benchmark_callable(
    fn: Callable[[], Any],
    warmup: int = 5,
    runs: int = 20,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """
    Benchmark any zero-argument callable.

    The callable should contain the operation we want to measure.

    Example:
        benchmark_callable(lambda: model(x))
    """

    if warmup < 0:
        raise ValueError("warmup must be >= 0.")

    if runs <= 0:
        raise ValueError("runs must be > 0.")

    # Warmup runs are not measured.
    for _ in range(warmup):
        fn()

    synchronize_if_needed(device)

    latencies_ms = []

    for _ in range(runs):
        start = time.perf_counter()

        result = fn()

        synchronize_if_needed(device)

        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies_ms.append(latency_ms)

        # Keep result referenced so Python does not optimize away the call pattern.
        _ = result

    total_time_ms = sum(latencies_ms)

    return {
        "warmup_runs": warmup,
        "measured_runs": runs,
        "avg_ms": total_time_ms / runs,
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "total_measured_ms": total_time_ms,
        "throughput_per_second": runs / (total_time_ms / 1000),
        "raw_latencies_ms": latencies_ms,
    }


def print_benchmark_summary(name: str, result: dict[str, Any]) -> None:
    """
    Print benchmark results in a readable format.
    """

    print(f"\n{name}")
    print("-" * len(name))

    print(f"Warmup runs: {result['warmup_runs']}")
    print(f"Measured runs: {result['measured_runs']}")
    print(f"Average latency: {result['avg_ms']:.6f} ms")
    print(f"Minimum latency: {result['min_ms']:.6f} ms")
    print(f"Maximum latency: {result['max_ms']:.6f} ms")
    print(f"p50 latency: {result['p50_ms']:.6f} ms")
    print(f"p95 latency: {result['p95_ms']:.6f} ms")
    print(f"p99 latency: {result['p99_ms']:.6f} ms")
    print(f"Throughput: {result['throughput_per_second']:.2f} ops/sec")