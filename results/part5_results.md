# Part 5 measured results

Measured on 2026-10-24 (America/Los_Angeles), RTX 5080, Ubuntu 24.04 WSL2.
The FP16 ResNet-18 engine from Part 2 is reused. Inputs come from the
actual DPDK pcap receiver after the packet validation fixes.

## Classification and reconstruction

| Received input | Byte-identical images | Matching classes | Maximum logit difference |
| --- | ---: | ---: | ---: |
| frames_rx_inorder.bin | 16/16 | 16/16 | 0 |
| frames_rx_messy.bin | 16/16 | 16/16 | 0 |
| frames_rx_shuffled.bin | 16/16 | 16/16 | 0 |

All 48 per-image comparisons passed. The received files and originals share
SHA-256 `1802d9b5f169d763f5c6d3b125826acdb819caa225f7dc7ab6457ce1d282bded`.
The images are synthetic; class agreement is not an accuracy measurement.

## Pinned-memory H2D copy

Each copy moves 16 x 224 x 224 x 3 uint8 values (2,408,448 bytes).
Each case uses 50 warmup and 200 timed copies. CUDA events cover only
the asynchronous pinned CPU-to-GPU copy, on one non-default stream.

| Input | Mean (ms) | Std (ms) | Effective GB/s |
| --- | ---: | ---: | ---: |
| frames_original | 0.047017 | 0.000513 | 51.225 |
| frames_rx_inorder | 0.044375 | 0.001086 | 54.275 |
| frames_rx_shuffled | 0.046740 | 0.006303 | 51.529 |
| frames_rx_messy | 0.044749 | 0.002916 | 53.821 |

File reading, allocation, CPU buffer fill, GPU preprocessing, FP16 conversion,
and inference are excluded. Identical input files can have different measured
copy times due to normal scheduling and measurement variation. These results
measure the staging copy, not an observed end-to-end GPUNetIO speedup.

## Preprocessing and validation

The full IMAGENET1K_V1 preset runs on GPU: NHWC-to-NCHW, resize to 256,
center crop to 224, scale, FP32 normalization, then contiguous FP16 conversion.
Original and reassembled paths use identical preprocessing and the same engine.

Packet regressions passed for both replay and actual DPDK, including malformed
chunks, valid retries, missing data, and verifier failure propagation.
GPU preprocessing matched analytic black/white inputs. A one-byte corruption
with unchanged class labels correctly failed byte validation; a truncated file
was rejected before inference. See [GPU checks](part5_validation.log).

The initial packet review is retained in [part34_review.md](part34_review.md);
its two findings are resolved by the current working tree.

Raw records: [classification CSV](part5_classification.csv), [H2D CSV](part5_h2d.csv),
[JSON](part5_summary.json), [classification log](part5.log),
[packet regressions](part34_regression.log), and [DPDK log](part34_validation.log).
Commands and scope are in the [Part 5 guide](../docs/part5.md).

Part 6 Docker/DOCA and Part 7 RDMA tests remain outside these results.
