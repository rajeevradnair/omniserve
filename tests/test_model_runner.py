from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.runtime.model_runner import ModelRunner


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        return self.linear(x)


def test_model_runner_returns_expected_shape():
    model = TinyClassifier()
    runner = ModelRunner(model, device="cpu")

    x = torch.randn(3, 4)

    logits = runner.predict(x)

    assert logits.shape == torch.Size([3, 2])


def test_model_runner_benchmark_returns_latency_fields():
    model = TinyClassifier()
    runner = ModelRunner(model, device="cpu")

    x = torch.randn(3, 4)

    result = runner.benchmark(x, warmup=1, runs=3)

    assert "avg_ms" in result
    assert "min_ms" in result
    assert "max_ms" in result
    assert result["avg_ms"] >= 0
