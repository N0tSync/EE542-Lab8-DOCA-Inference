# Directory migration validation

The functionality and original results were committed in `a836611` before the
layout change. The directory migration updates source locations, package imports,
Makefile paths, default output locations, and documentation links.

## Completed checks

- All Python sources compile; eight module help entry points exit successfully.
- C executables rebuild with `-Idpdk` and outputs under `build/`.
- All packet regression cases pass for replay and the actual DPDK pcap port,
  including both Makefile failure-propagation checks.
- Both TensorRT precisions pass dynamic-batch, output reuse, and stream tests.
- `make part5 SUDO= PART5_OUTPUT=results/validation/part5` completes the full
  generator -> real DPDK -> byte verification -> GPU classification pipeline.
- All three received datasets have 16/16 byte-identical images and matching
  classes (48 per-image comparisons total).
- Original raw logs, CSV/JSON results, generated image/PCAP data, and models
  retain their pre-migration SHA-256 hashes.
- Current Markdown links resolve and no Python/C sources or binary/PCAP data
  remain as files in the repository root.

## Evidence

- [Packet regressions](packet_regression.log)
- [TensorRT runner](runner.log)
- [Full pipeline output](pipeline.log)
- [Part 5 verification JSON](part5/part5_summary.json)
- [Preserved file hashes](preserved_files.json)

The verification run uses separate output files. The original Part 2 and Part 5
measurements remain under their respective result directories, including paths
recorded at the time of their original executions. No ONNX or engine rebuild
was needed for this directory-only change.
