"""Isolated, immediately terminable compute process; JSON lines on stdout."""

import json
import os
import sys
import time
from collections import deque

import psutil

from backend.config import RunConfig
from backend.engine import Engine
from backend.risk import StreamingRisk
from backend.telemetry import GIB


def emit(**data):
    print(json.dumps(data, allow_nan=False), flush=True)


def run(config: RunConfig):
    kinds = ["cpu", "gpu"] if config.engine == "compare" else [config.engine]
    for kind in kinds:
        engine = Engine(config, kind)
        risk = StreamingRisk(config.initial_value, config.confidence)
        cache = deque()
        cached_bytes = 0
        cache_limit = 0
        started = time.perf_counter()
        native_seconds = 0.0
        stride = len(config.assets) + 1 if config.memory_stress else 1
        # Fixed scratch ceiling, independent of total paths; reserve 128 MiB runtime overhead.
        batch_size = min(
            config.batch_size, max(1024, int((config.ram_gb * GIB * 0.25 - 128 * 1024**2) / (stride * 8 * 6)))
        )
        if kind == "gpu":
            # Keep individual launches short on display GPUs with Windows TDR.
            batch_size = min(
                batch_size,
                max(128, 2_000_000 // (config.steps * len(config.assets) ** 2)),
                int(config.vram_gb * GIB // (stride * 8)),
            )
        if config.memory_stress:
            # Use the requested budget for real scenarios while reserving runtime
            # and multiple batch-sized scratch buffers rather than idle filler.
            scratch_reserve = 512 * 1024**2 + batch_size * stride * 8 * 8
            cache_limit = max(0, int(config.ram_gb * GIB) - scratch_reserve)
        for start in range(0, config.paths, batch_size):
            n = min(batch_size, config.paths - start)
            if psutil.virtual_memory().available < config.reserve_gb * GIB + n * stride * 8 * 6:
                raise MemoryError("OS memory reserve would be breached by the next batch")
            t = time.perf_counter()
            batch = engine.batch(start, n, config.memory_stress)
            duration = time.perf_counter() - t
            native_seconds += duration
            risk.add(batch[:, 0])
            if cache_limit:
                while cache and cached_bytes + batch.nbytes > cache_limit:
                    cached_bytes -= cache.popleft().nbytes
                if batch.nbytes <= cache_limit:
                    cache.append(batch)
                    cached_bytes += batch.nbytes
            completed = start + n
            if kind == "gpu" and config.gpu_duty < 100:
                time.sleep(duration * (100 / config.gpu_duty - 1))
            elapsed = time.perf_counter() - started
            emit(
                type="progress",
                engine=kind,
                completed=completed,
                paths=config.paths,
                elapsed=elapsed,
                throughput=completed / elapsed,
                native_seconds=native_seconds,
                native_throughput=completed / native_seconds,
                cache_gb=cached_bytes / GIB,
                results=risk.result(),
            )
        # A bounded, genuine sample of retained per-asset scenarios, not generated chart data.
        retained_sample = cache[-1][:8].tolist() if cache else []
        cache_paths = sum(len(chunk) for chunk in cache)
        asset_means = (sum((chunk[:, 1:].sum(axis=0) for chunk in cache)) / cache_paths).tolist() if cache_paths else []
        emit(
            type="engine_complete",
            engine=kind,
            completed=config.paths,
            elapsed=time.perf_counter() - started,
            native_seconds=native_seconds,
            results=risk.result(),
            retained_sample=retained_sample,
            cache_gb=cached_bytes / GIB,
            cache_paths=cache_paths,
            cached_asset_mean_ratios=asset_means,
        )
        del cache, batch, engine
    emit(type="complete")


if __name__ == "__main__":
    try:
        # This is before compute. If a launcher is cancelled during startup,
        # the closed stdout pipe prevents an orphan interpreter doing work.
        emit(type="ready", pid=os.getpid())
        run(RunConfig.model_validate_json(sys.stdin.readline()))
    except Exception as exc:
        emit(type="error", error=f"{type(exc).__name__}: {exc}")
        sys.exit(1)
