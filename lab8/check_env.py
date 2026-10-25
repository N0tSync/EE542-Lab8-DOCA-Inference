"""Record the Part 1 environment and run GPU and DPDK development checks."""

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile

import tensorrt as trt
import torch


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        return {"command": shlex.join(args), "returncode": result.returncode,
                "output": (result.stdout + result.stderr).strip()}
    except OSError as error:
        return {"command": shlex.join(args), "returncode": 127, "output": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/environment/environment.json"))
    args = parser.parse_args()
    commands = {
        "nvidia_smi": ["nvidia-smi"], "nvcc": ["nvcc", "--version"],
        "dpdk": ["pkg-config", "--modversion", "libdpdk"],
        "libpcap": ["pkg-config", "--modversion", "libpcap"],
        "libibverbs": ["pkg-config", "--modversion", "libibverbs"],
        "pip_check": [sys.executable, "-m", "pip", "check"],
        "compiler": ["g++", "--version"],
        "system_packages": ["dpkg-query", "-W", "-f=${Package} ${Version}\n", "build-essential",
                            "pkg-config", "dpdk", "dpdk-dev", "libdpdk-dev", "libpcap-dev",
                            "rdma-core", "libibverbs-dev", "ibverbs-utils", "cuda-toolkit-13-0"],
    }
    checks = {name: command(cmd) for name, cmd in commands.items()}
    with tempfile.TemporaryDirectory() as temp:
        source, binary = Path(temp) / "dpdk_check.c", Path(temp) / "dpdk_check"
        source.write_text('#include <stdio.h>\n#include <rte_version.h>\nint main(void) { puts(rte_version()); return 0; }\n')
        flags = command(["pkg-config", "--cflags", "--libs", "libdpdk"])
        if flags["returncode"] == 0:
            checks["dpdk_compile"] = command(["cc", str(source), "-o", str(binary), *shlex.split(flags["output"])])
            if checks["dpdk_compile"]["returncode"] == 0:
                checks["dpdk_link_run"] = command([str(binary)])
    passed = all(item["returncode"] == 0 for item in checks.values())
    gpu = {}
    try:
        x = torch.randn(512, 512, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(y).all())
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        gpu = {"name": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
               "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
               "matmul_finite": finite, "tensorrt_builder": builder is not None}
        passed = passed and finite and builder is not None
    except Exception as error:
        gpu["error"] = str(error)
        passed = False
    report = {
        "timestamp": datetime.now().astimezone().isoformat(), "python": sys.version,
        "python_executable": sys.executable, "kernel": platform.release(),
        "os_release": platform.freedesktop_os_release(), "cuda_runtime": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "tensorrt-cu13", "onnx", "onnxscript", "numpy", "scapy")},
        "tools": {name: shutil.which(name) for name in ("nvcc", "cc", "pkg-config", "meson", "ninja")},
        "gpu": gpu, "checks": checks, "passed": passed,
        "scope": "Part 1 development environment; DPDK packet RX and DOCA are separate team tasks",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
