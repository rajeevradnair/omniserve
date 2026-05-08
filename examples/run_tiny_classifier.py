from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.runtime.model_runner import ModelRunner
from src.runtime.tensor_inspector import inspect_tensor


class TinyClassifier(nn.Module):
    """
    Tiny classifier used to demonstrate ModelRunner.

    Input shape:
        [batch_size, 4]

    Output shape:
        [batch_size, 2]
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        logits = self.linear(x)
        return logits


def postprocess_logits(logits: torch.Tensor) -> dict:
    """
    Convert raw logits into probabilities and predicted class.
    """

    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

    return {
        "logits": logits,
        "probabilities": probabilities,
        "predicted_class": predicted_class,
    }


def main():
    model = TinyClassifier()
    runner = ModelRunner(model)

    x = torch.tensor(
        [
            [0.5, 1.0, -0.2, 0.7],
            [1.2, -0.4, 0.3, 2.1],
        ],
        dtype=torch.float32,
    )

    inspect_tensor("input_batch", x)

    logits = runner.predict(x)

    inspect_tensor("output_logits", logits)

    result = postprocess_logits(logits)

    print("\nProbabilities:")
    print(result["probabilities"])

    print("\nPredicted classes:")
    print(result["predicted_class"])

    benchmark_result = runner.benchmark(x, warmup=5, runs=20)

    print("\nBenchmark result:")
    for key, value in benchmark_result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
