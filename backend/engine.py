"""C ABI bridge. CPU and CUDA use the same path-indexed random stream."""

import ctypes as ct
import os
import shutil
from pathlib import Path

import numpy as np

from backend.config import RunConfig

ROOT = Path(__file__).resolve().parents[1]
_dll_dirs = []
if os.name == "nt":
    compiler = shutil.which("g++")
    if compiler:
        _dll_dirs.append(os.add_dll_directory(str(Path(compiler).parent)))


class Parameters(ct.Structure):
    _fields_ = [
        ("assets", ct.c_int),
        ("steps", ct.c_int),
        ("horizon", ct.c_double),
        ("seed", ct.c_uint64),
        ("weights", ct.c_double * 16),
        ("mu", ct.c_double * 16),
        ("sigma", ct.c_double * 16),
        ("chol", ct.c_double * 256),
    ]


def library_path(engine: str) -> Path:
    suffix = ".dll" if os.name == "nt" else ".so"
    return ROOT / "build" / (f"qf_{engine}" + suffix)


class Engine:
    def __init__(self, config: RunConfig, kind: str):
        self.config = config
        self.kind = kind
        path = library_path(kind)
        if not path.exists():
            raise RuntimeError(
                f"{kind.upper()} library missing; run python scripts/build.py" + (" --cuda" if kind == "gpu" else "")
            )
        self.lib = ct.CDLL(str(path))
        self.lib.qf_simulate.argtypes = [
            ct.POINTER(Parameters),
            ct.c_uint64,
            ct.c_uint64,
            ct.c_int,
            ct.c_int,
            ct.POINTER(ct.c_double),
        ]
        self.lib.qf_simulate.restype = ct.c_int
        if kind == "gpu" and not self.lib.qf_gpu_available():
            raise RuntimeError("CUDA library found, but no CUDA device is available")
        self.p = Parameters()
        self.p.assets, self.p.steps, self.p.horizon, self.p.seed = (
            len(config.assets),
            config.steps,
            config.horizon,
            config.seed,
        )
        chol = np.linalg.cholesky(config.correlation)
        for a, asset in enumerate(config.assets):
            self.p.weights[a], self.p.mu[a], self.p.sigma[a] = asset.weight, asset.mu, asset.sigma
            for b in range(len(config.assets)):
                self.p.chol[a * 16 + b] = chol[a, b]

    def batch(self, start: int, count: int, retain_assets: bool = False) -> np.ndarray:
        stride = len(self.config.assets) + 1 if retain_assets else 1
        out = np.empty((count, stride), dtype=np.float64)
        threads = min(self.config.threads, max(1, int((os.cpu_count() or 1) * self.config.cpu_ceiling / 100)))
        resource = int(self.config.vram_gb * 1024) if self.kind == "gpu" else threads
        code = self.lib.qf_simulate(
            ct.byref(self.p), start, count, resource, stride, out.ctypes.data_as(ct.POINTER(ct.c_double))
        )
        if code:
            raise RuntimeError(f"Native engine error {code}; CUDA -2 means insufficient free VRAM")
        if not np.isfinite(out).all() or (out <= 0).any():
            raise RuntimeError("Non-finite/underflow scenario; reduce horizon or volatility")
        return out
