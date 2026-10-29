"""Small portable build entrypoint; CMake is an alternative, not a prerequisite."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action="store_true")
args = parser.parse_args()
(ROOT / "build").mkdir(exist_ok=True)
suffix = ".dll" if os.name == "nt" else ".so"
compiler = shutil.which("g++")
if not compiler:
    raise SystemExit("Install GCC with OpenMP (Windows: MSYS2 UCRT64 gcc), or use the CMake MSVC build in README.")
for kind in ("cpu", "research"):
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O3",
            "-march=native",
            "-fopenmp",
            "-shared",
            *([] if os.name == "nt" else ["-fPIC"]),
            str(ROOT / ("native/" + kind + ".cpp")),
            "-o",
            str(ROOT / "build" / ("qf_" + kind + suffix)),
        ],
        check=True,
    )
if args.cuda:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise SystemExit(
            "CPU built. CUDA Toolkit nvcc unavailable. Install Toolkit plus supported MSVC host compiler; then retry --cuda."
        )
    subprocess.run(
        [
            nvcc,
            "-std=c++20",
            "-O3",
            "-shared",
            *([] if os.name == "nt" else ["-Xcompiler", "-fPIC"]),
            str(ROOT / "native/gpu.cu"),
            "-o",
            str(ROOT / "build" / ("qf_gpu" + suffix)),
        ],
        check=True,
    )
print("Native build complete")
