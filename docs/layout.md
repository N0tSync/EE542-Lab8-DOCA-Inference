# Repository layout

Run commands from the repository root with the lab Python environment active.
Python modules use the `lab8` package; no editable installation or `PYTHONPATH`
configuration is required.

| Directory | Contents | Version control |
| --- | --- | --- |
| `lab8/` | Python environment, inference, packet, and integration modules | Tracked |
| `dpdk/` | DPDK receiver and shared C reassembly header | Tracked |
| `tests/` | GPU/packet checks and the standalone C replay helper | Tracked |
| `docs/` | Environment, interfaces, commands, and packet format | Tracked |
| `results/` | Logs, CSV/JSON measurements, and reports grouped by part | Tracked |
| `build/` | Compiled receiver and replay executables | Generated, ignored |
| `models/` | Pretrained weights, ONNX files, engines, and cache | Generated, ignored |
| `data/pcaps/` | Generated ordered, shuffled, and mixed PCAPs | Generated, ignored |
| `data/frames/` | Original and reconstructed raw RGB files | Generated, ignored |
| `data/previews/` | Optional PNG previews | Generated, ignored |
| `data/cifar10/` | Optional downloaded image source | Generated, ignored |

## Module and Makefile entry points

```bash
python -m lab8.check_env
python -m lab8.inference.export_onnx
python -m lab8.inference.build_engine
python -m lab8.inference.benchmark
python -m lab8.packets.generate_pcap
python -m tests.test_packet_pipeline --dpdk
python -m tests.test_trt_runner
make part5 SUDO=
```

The root Makefile builds C executables under `build/`, generates PCAPs under
`data/pcaps/`, writes RGB buffers under `data/frames/`, and invokes the Part 5
module. Its existing `all`, `pcap_replay_test`, `test_rx`, `run`, and `part5`
targets remain available. `SUDO=` is optional for hosts allowing unprivileged
DPDK pcap access. `PYTHON` and `EAL` remain configurable.

Use explicit package imports when reusing the runner:

```python
from lab8.inference.trt_runner import TensorRTRunner
```

## Results and migration

Environment snapshots are in `results/environment/`; Part 2, Parts 3-4, and
Part 5 records are in `results/part2/`, `results/part34/`, and `results/part5/`.
The directory migration preserves the raw logs, CSVs, JSONs, image data, and
models byte for byte. Paths embedded in historical records describe their
original execution and have not been rewritten. Markdown links and current
running instructions use the new layout.

Part 5 and the packet fixes were committed separately as `a836611` before the
directory migration. Results generated while checking the new layout are saved
under `results/validation/`, without overwriting the original measurements.
`make part5 PART5_OUTPUT=results/validation/part5 SUDO=` selects a separate
classification output directory.

See the [Part 2 guide](part2.md), [packet format](packet_format.md), and
[Part 5 guide](part5.md) for individual steps.
