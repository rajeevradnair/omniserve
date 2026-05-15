from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.runtime.model_runner import ModelRunner
from src.runtime.latency_timer import benchmark_callable, print_benchmark_summary
from src.runtime.tensor_inspector import inspect_tensor


class TinyClassifier(nn.Module):
    """
    Tiny classifier used for latency benchmarking.

    Input shape:
        [batch_size, 4]

    Output shape:
        [batch_size, 2]
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        return self.linear(x)


def run_batch_benchmark(batch_size: int) -> dict:
    """
    Benchmark TinyClassifier for a given batch size.
    """

    model = TinyClassifier()
    runner = ModelRunner(model)

    x = torch.randn(batch_size, 4, dtype=torch.float32)

    moved_x = runner.move_inputs_to_device(x)

    inspect_tensor(f"input_batch_size_{batch_size}", moved_x)

    result = benchmark_callable(
        fn=lambda: runner.predict(moved_x),
        warmup=10,
        runs=50,
        device=runner.device,
    )

    result["batch_size"] = batch_size

    return result


def main():
    batch_sizes = [1, 2, 4, 8, 16, 32]

    all_results = []

    for batch_size in batch_sizes:
        result = run_batch_benchmark(batch_size)
        all_results.append(result)

        print_benchmark_summary(
            name=f"TinyClassifier batch_size={batch_size}",
            result=result,
        )

    print("\nCompact comparison")
    print("------------------")
    print("batch_size, avg_ms, p95_ms, throughput_per_second")

    for result in all_results:
        print(
            f"{result['batch_size']}, "
            f"{result['avg_ms']:.6f}, "
            f"{result['p95_ms']:.6f}, "
            f"{result['throughput_per_second']:.2f}"
        )


if __name__ == "__main__":
    main()