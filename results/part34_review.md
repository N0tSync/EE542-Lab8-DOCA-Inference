# Parts 3-4 review

Reviewed commit: `3b864f6` on 2026-10-24 (America/Los_Angeles).
Follow-up: both findings below have been fixed in the working tree. The
[regression checks](part34_regression.log) and [real DPDK rerun](part34_validation.log)
passed. The original findings and logs are retained as a record of the reviewed commit.

The submitted source files were copied unchanged to a temporary workspace for
compilation and execution. No member B source files were modified.

## Scope and result

The normal Part 3 and Part 4 scenarios work, including real DPDK pcap-port
execution, not only the standalone replay helper. The raw RGB layout matches
the Part 5 handoff: 16 x 224 x 224 x 3 uint8 HWC, 2,408,448 bytes total.
Two reproducible validation defects remain before final sign-off.

| Input | Received | Accepted | Duplicate drops | Other-port drops | Complete images | Byte comparison |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| In order | 1728 | 1728 | 0 | 0 | 16 | 16/16 identical |
| Shuffled | 1728 | 1728 | 0 | 0 | 16 | 16/16 identical |
| Shuffled + 50 duplicates + 20 other-port packets | 1798 | 1728 | 50 | 20 | 16 | 16/16 identical |

The standalone replay helper and the actual DPDK receiver both produced outputs
identical to the originals. Independent Python byte equality and `cmp` confirmed
the DPDK results without relying on the Makefile verification pipeline.
All three generated PCAPs have a maximum Ethernet frame size of 1464 bytes,
corresponding to 1450-byte IPv4 packets within the 1500-byte MTU.

Removing one data chunk correctly resulted in 15 complete images, an incomplete
frame 0 (107/108 chunks), idle-timeout termination, and receiver exit status 1.

## Finding 1: Makefile verification failures are masked

Location: `Makefile`, lines 24 and 30.

Both recipes pipe the verifier into `tail -1`. Under the default make shell,
the pipeline status comes from `tail`, so a failing verifier can still make the
target succeed.

Reproduction used the unmodified Makefile and an isolated copy of the generated
PCAPs. One byte of `frames_original.bin` was changed, then
`make -o pcaps test_rx` was run (`-o pcaps` preserves that controlled mismatch).
All three comparisons printed `15/16 images identical byte for byte`, but make
returned 0. The standalone verifier correctly returns 1 on those mismatches.

Recommended correction: run the verifier directly in both recipes, or explicitly
preserve its exit status. Removing the `tail` pipeline is the simplest option.

## Finding 2: malformed chunks can be reported as a complete image

Location: `rx_core.h`, lines 76-93 and 97-102.

The receiver checks bounds and distinct sequence IDs, but not that their ranges
cover the entire frame. A zero-length chunk or a chunk overlapping another
range is still marked as received. `got == nseg` then declares completion.

Two separately generated PCAPs changed only frame 0, chunk 1: one set its offset
to zero, and one set its declared length to zero. Network headers and checksums
were rebuilt. In both cases, the replay helper and actual DPDK receiver returned
0 and reported 16 complete images with zero drops. Independent verification
returned 1 and found only 15/16 images identical.

Recommended correction: for the documented fixed 1400-byte protocol, validate
frame size, segment count, `offset == seq * 1400`, and the exact expected chunk
length (including the 728-byte final chunk). Reject empty or inconsistent chunks
before marking their sequence IDs received. Alternatively, track byte coverage.

## Execution details and evidence

The unchanged sources compiled using the installed DPDK 23.11.4 and libpcap.
The baseline command was `make all test_rx`. Real DPDK runs used:

```bash
./dpdk_rx -l 0 --no-huge -m 512 --no-pci \
  --vdev 'net_pcap0,rx_pcap=images_inorder.pcap,tx_pcap=/dev/null' \
  -- frames_dpdk_inorder.bin 16
python3 verify_frames.py frames_original.bin frames_dpdk_inorder.bin
cmp frames_original.bin frames_dpdk_inorder.bin
```

The same commands were repeated with the shuffled and messy PCAPs. Malformed
PCAPs were generated with the submitted manual packet builder for focused
validation; the normal baseline used Scapy. A 20-second external timeout bounded
each receiver invocation.

The agent sandbox blocks Scapy interface discovery and DPDK runtime directory
creation, so those checks ran with approved host access. WSL's automatic TAP
probes emitted permission warnings; the requested pcap port still initialized,
received the expected packets, and passed independent byte comparisons.

Raw output and the structured test summary are in
[`part34_review.log`](part34_review.log). PCAPs and binary outputs remain in the
temporary workspace `/tmp/ee542-lab8-partb-review.SVAs8n`.
These review records have not been committed or pushed.
