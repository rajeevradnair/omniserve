from __future__ import annotations
import time
from typing import Any

import torch
import torch.nn as nn


class ModelRunner:
    def __init__(self, model: nn.Module, device: torch.device | str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    def move_inputs_to_device(self, inputs: Any) -> Any:
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
                else:
                    inputs[k] = v
        elif isinstance(inputs, list):
            return [self.move_inputs_to_device(item) for item in inputs]
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")

    def predict(self, inputs: Any) -> Any:
        inputs = self.move_inputs_to_device(inputs)
        with torch.no_grad():
            outputs = (
                self.model(**inputs) if isinstance(inputs, dict) else self.model(inputs)
            )
        return outputs

    def benchmark(
        self, inputs: Any, warmup: int = 5, runs: int = 20
    ) -> dict[str, float | int | str]:
        moved_inputs = self.move_inputs_to_device(inputs)
        
        with torch.no_grad():
            for _ in range(warmup):
                if isinstance(moved_inputs, dict):
                    _ = self.model(**moved_inputs)
                else:
                    _ = self.model(moved_inputs)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            runtimes_ms = []
            with torch.no_grad():
                for _ in range(runs):
                    
                    start = time.perf_counter()

                    if isinstance(moved_inputs, dict):
                        _ = self.model(**moved_inputs)
                    else:
                        _ = self.model(moved_inputs)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

                    end = time.perf_counter()
                    runtime = (end - start) * 1000
                    runtimes_ms.append(runtime)

        avg_runtime = sum(runtimes_ms) / len(runtimes_ms)
        return {
            "average_runtime_ms": avg_runtime,
            "warmup_runs": warmup,
            "measured_runs": runs,
            "device": str(self.device),
            "avg_ms": avg_runtime,
            "min_ms": min(runtimes_ms),
            "max_ms": max(runtimes_ms),
        }
