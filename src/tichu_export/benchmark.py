"""Sequential single-input latency benchmark.

Measures per-call latency of `module(*inputs)` over `n` iterations and
reports p50, p95, p99, and mean in milliseconds. A 10-iter warmup is
discarded so the first-call JIT/cache cost does not bias the tail.
"""

import time

import numpy as np
import torch


_WARMUP = 10


def benchmark_p99(module, *example_inputs, n: int = 1000) -> dict[str, float]:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    module = module.eval() if isinstance(module, torch.nn.Module) else module

    with torch.no_grad():
        for _ in range(_WARMUP):
            module(*example_inputs)

        timings_ms: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            module(*example_inputs)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(timings_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }
