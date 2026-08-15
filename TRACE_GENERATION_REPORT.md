# A6000 Trace Generation Report

## Objective

Generate and validate remodeled benchmark traces on the NVIDIA RTX A6000, then
move (not copy) each suite's workload directories from
`simulator-remodeled/hw_run/traces/device-0/12.8` into its corresponding
`A6000_dataset` directory.

## Status

| Suite | Status | Notes |
|---|---|---|
| `rodinia_2.0-ft` | Complete/archived | 10 workloads; validated archive retained and original dataset removed to save space. |
| `rodinia-3.1` | Complete/archived | 20 workloads / 34 runs / 6,955 launches; validated archive retained and original dataset removed. |
| `parboil` | Complete/archived | 8 workloads / 8 runs / 250 launches; validated archive retained and original dataset removed. |
| `polybench` | Complete/archived | 11 workloads / 11 runs / 867 launches; validated archive retained and original dataset removed. |
| `ispass-2009` | Complete/archived | 7 enabled workloads / 7 runs / 19 launches; validated archive retained and original dataset removed. |
| `pannotia` | Complete/archived | 8 workloads / 13 runs; 942 launches; validated archive retained and original dataset removed to save space. |
| `lonestargpu-2.0` | Complete/archived | 6 applications / 19 runs; 676 launches; validated archive retained and original dataset removed to save space. |
| `proxy-apps-doe` | Complete/archived | 3 workloads / 38 launches; validated archive retained and original dataset removed to save space. |
| `dragon-naive` | Complete/archived | 4 applications / 8 runs / 164 launches; validated archive retained and original dataset removed to save space. |
| `dragon-cdp` | Complete/archived | 2 applications / 6 runs / 145 tracer-visible parent launches; validated archive retained and original dataset removed to save space. |
| `GPU_Microbenchmark` | Complete/archived | 15 applications / 15 runs / 15 launches; validated archive retained and original dataset removed to save space. |
| `Deepbench_nvidia_tencore_gemm` | Complete/archived | 1 application / 9 runs / 78 launches; mixed-architecture and JIT SASS retained; validated archive retained and original dataset removed. |
| `cutlass_5_trace` | Complete/archived | 20 runs; one timed launch retained per run with complete SASS/CUBIN/RFU; validated archive retained and raw suite removed. |

## Source and Configuration Adjustments

### Rodinia 3.1 CFD

- Original behavior: 2000 solver iterations and 14,003 dynamic kernel launches
  per input.
- Adjustment: `euler3d.cu` reads `RODINIA_CFD_ITERATIONS`; the Rodinia 3 trace
  script defaults it to 100.
- Expected trace size: approximately 703 kernel launches per CFD input.
- Standalone behavior without the environment variable remains 2000 iterations.
- Validation: all three configured inputs completed with exactly 703 dynamic
  launches each; the enhanced tracer parsed all 4/4 static kernels for each
  input and preserved the intermediate SASS artifacts.

### Rodinia 3.1 LavaMD

- Original input: `-boxes1d 10`, which creates 1000 boxes and is annotated as
  requiring 130 GB in the suite definition.
- Problem on this host: the A6000 trace machine has 56 GiB RAM. During the
  single dynamic kernel, tracer RSS grew linearly past 13 GiB before any trace
  could be committed, making an OOM failure likely.
- Adjustment: changed both the authoritative suite YAML and LavaMD's `run`
  file to `-boxes1d 4`. The box count is reduced from 1000 to 64 while keeping
  the same kernel and neighbor-interaction code path.
- Validation: the reduced run completed in 61.7 seconds with one dynamic
  kernel; the enhanced tracer parsed 1/1 static kernel and preserved its SASS.

### Rodinia 3.1 Myocyte

- Input fix: replaced hard-coded `../../data/myocyte/{y.txt,params.txt}` paths
  with the tracer run directory's `./data/{y.txt,params.txt}` symlink paths.
- Iteration adjustment: changed the trace input from `100 1 0` to `20 1 0`.
  With valid input, the original range was already above 750 launches while
  still running; the reduced range completed with 780 dynamic launches.
- Validation: both data files loaded without warnings, the run completed in
  about 190 seconds, and the enhanced tracer parsed 1/1 static kernel.

### Rodinia 3.1 SRAD v1 input

- Input fix: changed the hard-coded `../../../data/srad/image.pgm` path to
  `./data/image.pgm` and added the missing `srad_v1-rodinia-3.1` data-directory
  mapping to the existing SRAD data.
- The corrected run no longer reports a failed image read.

### Rodinia 3.1 Streamcluster

- The small input (`3 6 16 65536 65536 1000 none output.txt 1`) completed with
  224 dynamic launches.
- The high-dimensional input (`10 20 256 65536 65536 1000 none output.txt 1`)
  completed with 1,611 dynamic launches in about 21,014 seconds and produced
  approximately 128 GiB of trace data by itself.
- This is a data-dependent clustering/convergence path, rather than a fixed
  14,000-iteration loop like the original CFD configuration.  The full
  original input was retained because it completed successfully, but it is
  classified as a heavy workload.  For future faster trace regeneration, the
  second input should be replaced by a smaller dimension/point-count input,
  rather than truncating a trace in the middle of convergence.

### PolyBench Gram-Schmidt

- Original behavior: 2048 decomposition iterations and 6144 dynamic launches.
- Adjustment: CPU reference and GPU execution both read
  `POLYBENCH_GRAMSCHMIDT_ITERATIONS`; the trace script defaults it to 100.
- Expected trace size: 300 launches, while retaining CPU/GPU validation.

### PolyBench FDTD-2D

- Original behavior: 500 time steps and 1500 dynamic launches.
- Adjustment: CPU reference and GPU execution both read
  `POLYBENCH_FDTD_ITERATIONS`; the trace script defaults it to 100.
- Expected trace size: 300 launches, while retaining CPU/GPU validation.

## Problems and Resolutions

### Rodinia 3.1 SRAD v2 link failure

- Symptom: linker could not find `-lcutil_x86_64` with CUDA 12.8.
- Cause: legacy Rodinia build logic unconditionally expected the removed CUDA
  SDK cutil library.
- Resolution: updated the Rodinia common build configuration to avoid linking
  the obsolete library when it is unavailable.

### Parboil SGEMM illegal instruction

- Symptom: SGEMM exited with `Illegal instruction` immediately after reading
  its first matrix, including during a native run without NVBit.
- Cause: the successful path of its `bool` matrix-read function did not return
  a value, causing undefined behavior with the current compiler.
- Resolution: added the missing `return true`, rebuilt SGEMM, verified a native
  run, and regenerated the trace successfully.

### Parboil SPMV historical crash note

- The suite YAML contains the old inline note `crash not kernelslist.g`, but
  SPMV completed successfully on the current A6000/CUDA 12.8/NVBit setup.
- The retained result contains 50 launches of one static kernel, a valid
  dynamic protobuf, parseable enhanced JSON, SASS, and CUBIN.

### ISPASS-2009 disabled WP build failure

- Symptom: the aggregate ISPASS build failed because disabled workload WP
  invoked an unavailable `gfortran` compiler.
- Cause: LIB and WP are commented out in the trace-suite YAML, but the legacy
  aggregate make target still attempted to compile both.
- Resolution: added `ISPASS_ENABLED_ONLY=1` to align the build with the seven
  workloads actually enabled in the trace configuration. The trace script now
  uses that mode and does not require the unrelated WP Fortran dependency.

### Rodinia 3.1 LavaMD trace-memory exhaustion risk

- Symptom: the original `boxes1d=10` run buffered an ever-growing single-kernel
  trace in RAM; RSS reached about 14 GiB after 335 seconds and was still rising.
- Cause: the number of boxes scales as `boxes1d^3`; the supplied suite metadata
  explicitly budgets 130 GB, exceeding this host's 56 GiB RAM.
- Resolution: interrupted only the invalid LavaMD run before OOM, removed its
  24 KB/zero-kernel partial directory, reduced `boxes1d` to 4, and resumed from
  LavaMD using a recovery-only suite so completed Rodinia traces were not
  regenerated or overwritten.

### Rodinia 3.1 Myocyte and SRAD run-relative input paths

- Symptom: both programs printed `The file was not opened for reading` but
  returned success and generated traces.
- Cause: their source used paths relative to the original source-tree launch
  location, while the hardware tracer executes each input in an isolated run
  directory.  The programs continued with uninitialized input arrays.
- Resolution: changed both programs to use their generated `./data` symlinks,
  rebuilt Rodinia, removed the invalid partial traces (Myocyte: 156 launches;
  SRAD: 58 launches), and regenerated them.  Successful reruns are accepted
  only when the file-open warning is absent and enhanced SASS parsing passes.

### Rodinia 3.1 Streamcluster long-running input

- Symptom: the second configured input generated 1,611 launches and about
  128 GiB of trace data, taking approximately 5.84 hours.
- Cause: `pgain` launches the same static CUDA kernel repeatedly from a
  data-dependent clustering/local-search convergence loop.  The larger
  `kmin/kmax`, dimension, and point count make it substantially heavier than
  the first input; this is repeated launch of one static kernel, not 1,611
  different SASS kernels.
- Resolution for this dataset: allowed the original input to finish and kept
  its complete trace; enhanced tracing parsed 1/1 static kernel and retained
  its SASS/CUBIN.  The manifest labels it as a heavy original configuration.

### Legacy PolyBench direct Makefiles

- Symptom: direct per-workload Makefiles request unsupported architectures such
  as `compute_10` under CUDA 12.8.
- Resolution: project-level build configuration targets the A6000 (`sm_86`);
  modified Gram-Schmidt and FDTD sources were separately compile-verified for
  `sm_86`.

### PolyBench ATAX and GESUMMV uninitialized accumulators

- Symptom: the first traced ATAX run reported 4,096 mismatches, and GESUMMV
  reported 1,310 mismatches.
- Cause: both GPU implementations accumulated into `y/tmp` with `+=` after
  copying uninitialized host arrays. GESUMMV also left matrix `B` uninitialized.
- Resolution: zeroed the GPU accumulators with `cudaMemset`, initialized
  GESUMMV matrix `B`, rebuilt both executables, and verified native runs with
  zero mismatches. The invalid trace contents were removed and both workloads
  were regenerated successfully (ATAX 2/2 and GESUMMV 1/1 enhanced kernels).

### PolyBench SYRK tracer memory peak

- Symptom: the single SYRK kernel buffered a large number of instruction events
  before writing per-thread-block traces; tracer RSS peaked at about 26.6 GB.
- Resolution: monitored host memory and disk while retaining the original
  problem size. RSS stabilized with sufficient headroom, all 4,096 thread
  blocks were written, and the run completed with zero mismatches. No source
  reduction was needed.

### Pannotia legacy cutil link failure

- Symptom: the aggregate Pannotia build failed at BC with
  `cannot find -lcutil_x86_64` under CUDA 12.8.
- Cause: six legacy subproject Makefiles linked the CUDA SDK 4.2 cutil library,
  although none of the Pannotia source files uses a cutil API.
- Resolution: removed the obsolete SDK include/library paths and linked only
  the CUDA runtime. All Pannotia variants then built successfully.

### Pannotia BC all-source launch explosion

- Original behavior: BC runs a multi-level BFS and backtracking sequence for
  every graph vertex (1,024 and 2,048 sources for the configured inputs),
  producing many thousands of dynamic launches.
- Resolution: BC reads `PANNOTIA_BC_SOURCES`; the trace script defaults it to
  20, while standalone behavior remains full-size when the variable is absent.
  Both inputs completed with 141 launches and 4/4 enhanced static kernels.

### LonestarGPU graph-convergence trace growth

- BFS and SSSP graph traversals can require many data-dependent iterations on
  high-diameter road graphs. Added `LONESTAR_TRACE_MAX_ITERATIONS`; the trace
  script defaults it to 50, and `0` restores unlimited convergence.
- Added `LONESTAR_TRACE_SKIP_VERIFY`, defaulting to `1` in the trace script, to
  omit the large GPU verification pass after the core algorithm. Setting it to
  `0` restores the original behavior.
- The original SSSP worklist-node `rmat20.gr` run grew to approximately
  32.5 GiB RSS with only 21.1 GiB host memory remaining and had not completed.
  Its incomplete output was removed and that one configuration was replaced
  with bundled `rmat12.sym.gr`, which preserves the application and R-MAT graph
  family and completed in 14 recorded launches.
- The full per-input launch counts, capped runs, and mixed verification-pass
  provenance are recorded in `lonestargpu-2.0/TRACE_MANIFEST.md`.

### Proxy Apps DOE LULESH legacy cutil link failure

- Symptom: LULESH failed to link because the CUDA 4.2 common rules added
  `-lcutil_x86_64`, which CUDA 12.8 no longer provides.
- Source review confirmed LULESH does not call any cutil API. The shared rule
  now honors `OMIT_CUTIL_LIB` before adding the legacy dependency, and LULESH
  enables that switch. CUDA runtime linkage is unchanged.
- Kernel-count review: the configured `1e-7` stop time equals the initial time
  step, so LULESH naturally completed one iteration with 28 dynamic launches
  across 26 enhanced static kernels. No iteration reduction was needed.

### Dragon build dependencies and Join trace-memory growth

- Dragon's build initially failed because the host lacked its required
  `scons` command. Installed Ubuntu's SCons 4.5.2 package. The legacy build
  also requires `BOOST_ROOT`; both Dragon scripts now default it to `/usr` and
  validate the installed Boost headers.
- The original naive Join input (`204800/204800`) buffered one main kernel in
  host memory at roughly 4 GiB per minute. RSS reached 21.5 GiB with no kernel
  commit and was still growing; the suite's own 90 GB budget exceeds this
  56 GiB host. The incomplete run was stopped and removed before OOM.
- Reduced both Join input counts by 10x to 20,480. The same algorithm stages
  completed with 9 launches, produced 409,578 outputs, passed verification,
  and parsed 6/6 static kernels. All other Dragon naive inputs were retained.

### Dragon CDP CUDA 12.8 metadata and SSSP launch saturation

- CUDA 12.8 internal helpers `__cudaCDP2GetParameterBufferV2`,
  `__cudaCDP2LaunchDeviceV2`, and `__cudaCDP2StreamCreateWithFlags` contain
  parsed SASS but no RFU record. The enhanced tracer previously inserted a
  duplicate no-binary fallback and aborted. It now keeps parsed SASS when only
  RFU is absent; all final BFS and SSSP runs parsed 8/8 and 6/6 static kernels.
- coPapersDBLP has 283,547 vertices of degree at least 32. The original SSSP
  CDP threshold of 32 could exceed the application's 131,072 pending child
  launch limit, while its device code ignored launch failures, producing
  non-deterministic `CDP Incorrect!` results.
- Raised only the SSSP CDP threshold to 256. The graph still has 11,112
  CDP-eligible vertices, so child launches remain active. Also set the
  coPapers SSSP frontier scale to 2. The rebuilt binary passed five consecutive
  native checks and the final traced CPU-reference check.
- The tracer's 145 stats rows cover host-visible parent/contract kernels. CDP
  child grids are present as static SASS but are not independent `stats.csv`
  rows, so 145 must not be interpreted as the total hardware child-launch
  count.
- Final archive: `dragon-cdp.tar.zst`, 407,037,113 bytes, SHA-256
  `8b6eb9b30aa8715f79b7630619cf594482b5a71be09aa483518e7a5a983f8377`.
  It passed `zstd -t` and a tar-member check before the raw directory was
  removed.

### GPU Microbenchmark trace interpretation

- All 15 enabled microbenchmarks completed with one launch and one enhanced
  static kernel each; 30 SASS and 30 CUBIN files were retained in the archive.
- The suite definition comments out `l2_bw_128`; three atomic programs are
  built by the aggregate target but are not members of this trace suite.
- Printed bandwidth/latency values include NVBit instrumentation overhead and
  should not be treated as native A6000 performance measurements.
- Final archive: `GPU_Microbenchmark.tar.zst`, 36,551,075 bytes, SHA-256
  `5ed864ed1b6dc67ac9d7805ce13ded1262bd953ecff329d8b4a7a4179a77c627`.

## Validation Policy

A suite is marked complete only after its script exits successfully, every
configured workload run has a trace directory, dynamic kernel counts are
checked for obvious truncation or explosion, and the workload directories have
been moved to the corresponding dataset directory.

## Rodinia 3.1 Final Validation and Move

- Script/recovery runner exit status: success.
- Workload directories: 20.
- Configured run directories: 34.
- Non-empty `dynamic_trace.pb`: 34/34.
- Non-empty `stats.csv`: 34/34.
- Parseable `enhanced_execution_info.json`: 34/34.
- Runs containing non-empty SASS and CUBIN: 34/34.
- SASS files / CUBIN files: 67 / 67.
- Total dynamic launches: 6,955.
- Dataset size: approximately 286 GiB.
- Move verification: 20 directories are present under
  `A6000_dataset/rodinia-3.1`; zero `*-rodinia-3.1` directories remain under
  the tracer output root.
- Detailed per-input counts are recorded in
  `rodinia-3.1/TRACE_MANIFEST.md`.

## Parboil Final Validation and Move

- Script exit status after the SGEMM fix: success.
- Workload directories / configured runs: 8 / 8.
- Non-empty `dynamic_trace.pb` and `stats.csv`: 8/8.
- Parseable `enhanced_execution_info.json`: 8/8.
- Runs containing non-empty SASS and CUBIN: 8/8.
- SASS files / CUBIN files: 15 / 15.
- Total dynamic launches: 250 across 14 distinct launched kernel names.
- Dataset size: 12,321,388,843 bytes (approximately 11.5 GiB).
- Kernel-count review: maximum 100 launches (`stencil`); no unreasonable
  thousands-scale workload required reduction.
- Move verification: eight `parboil-*` directories are present under
  `A6000_dataset/parboil`; zero remain in the tracer output root.
- Detailed per-workload counts are recorded in `parboil/TRACE_MANIFEST.md`.

## Parboil Compressed Copy

- Archive: `A6000_compressed_dataset/parboil.tar.zst`
- Archive size: 328,154,011 bytes (approximately 313 MiB)
- Source dataset size: 12,321,388,843 bytes (approximately 11.5 GiB)
- Compressed/source ratio: approximately 2.66% (37.5x smaller)
- SHA-256: `045d6c98264d57a726a966c606cb560eafda77dcce6dc1701dbafeb2155bdd2e`
- Validation: the archive was first written as `.partial`, passed `zstd -t`,
  and was then renamed to its final name. The source dataset remains present.

## ISPASS-2009 Final Validation and Move

- Script exit status after the enabled-only build fix: success.
- Enabled workload directories / configured runs: 7 / 7.
- Non-empty `dynamic_trace.pb` and `stats.csv`: 7/7.
- Parseable `enhanced_execution_info.json`: 7/7.
- Runs containing non-empty SASS and CUBIN: 7/7.
- SASS files / CUBIN files: 8 / 8.
- Total dynamic launches: 19 across 10 distinct launched kernel names.
- Dataset size: 590,990,055 bytes (approximately 564 MiB).
- Kernel-count review: maximum 10 launches (BFS); no reduction was needed.
- Move verification: seven `ispass-2009-*` directories are present under
  `A6000_dataset/ispass-2009`; zero remain in the tracer output root.
- Detailed counts and the disabled-workload scope are recorded in
  `ispass-2009/TRACE_MANIFEST.md`.

## ISPASS-2009 Compressed Copy

- Archive: `A6000_compressed_dataset/ispass-2009.tar.zst`
- Archive size: 30,809,824 bytes (approximately 30 MiB)
- Source dataset size: 590,990,055 bytes (approximately 564 MiB)
- Compressed/source ratio: approximately 5.21% (19.2x smaller)
- SHA-256: `d41d663eb8541238a4e80202d9ac60fe9ecd8bdbe5fe1cd9e47882850f7f54da`
- Validation: the archive was first written as `.partial`, passed `zstd -t`,
  and was then renamed to its final name. The source dataset remains present.

## PolyBench Final Validation and Move

- Script exit status: success; corrected ATAX/GESUMMV reruns also exited
  successfully with zero CPU/GPU mismatches.
- Workload directories / configured runs: 11 / 11.
- Non-empty `dynamic_trace.pb` and `stats.csv`: 11/11.
- Parseable `enhanced_execution_info.json`: 11/11.
- Runs containing non-empty SASS and CUBIN: 11/11.
- SASS files / CUBIN files: 22 / 22; 11 SASS modules contain application
  function disassembly and 11 are CUDA runtime stub modules.
- Total dynamic launches: 867 across 20 distinct launched kernel names.
- Dataset apparent size: 37,394,607,438 bytes (approximately 34.8 GiB);
  allocated size is approximately 51 GiB due to small trace files.
- Kernel-count review: FDTD-2D and Gram-Schmidt are intentionally limited to
  300 launches each. 3DConvolution's 254 launches are repeated execution of one
  static kernel across 254 interior slices and are expected.
- Move verification: eleven `polybench-*` directories are present under
  `A6000_dataset/polybench`; zero remain in the tracer output root.
- Detailed counts and fixes are recorded in `polybench/TRACE_MANIFEST.md`.

## PolyBench Compressed Copy

- Archive: `A6000_compressed_dataset/polybench.tar.zst`
- Archive size: 1,165,203,652 bytes (approximately 1.09 GiB)
- Source apparent size: 37,394,607,438 bytes (approximately 34.8 GiB)
- Compressed/source ratio: approximately 3.12% (32.1x smaller)
- SHA-256: `f61a7d29a6d861fc1ebbcfcb4bbdc1957025b0921ce3a7e6d81986444bdefaa0`
- Validation: the archive was first written as `.partial`, passed `zstd -t`
  against its complete 45,916,784,640-byte tar stream, and was then renamed to
  its final name. The archive has the expected `polybench/` top-level path and
  the source dataset remains present with all 11 workload directories.

## Pannotia Final Validation and Archived Storage

- Script exit status after build and BC fixes: success.
- Workload directories / configured runs: 8 / 13.
- Non-empty dynamic protobuf and stats files: 13/13.
- Parseable enhanced JSON and retained SASS/CUBIN: 13/13.
- SASS files / CUBIN files: 26 / 26.
- Total dynamic launches: 942.
- Dataset apparent size before compression: 14,902,578,133 bytes
  (approximately 13.9 GiB).
- Archive: `A6000_compressed_dataset/pannotia.tar.zst`
- Archive size: 1,660,242,012 bytes (approximately 1.55 GiB)
- SHA-256: `339ab8cebf645ceb7399b2a9c3e79e08ca707ff60739e0c4fbb1a75ea8da6b90`
- Validation: the `.partial` archive passed complete `zstd -t` validation
  against a 17,139,343,360-byte tar stream before being renamed.
- Space policy: after archive validation, `A6000_dataset/pannotia` was deleted;
  its detailed `TRACE_MANIFEST.md` remains inside the archive.

## LonestarGPU 2.0 Final Validation and Archived Storage

- Generation/recovery exit status: success.
- Application directories / configured runs: 6 / 19.
- Non-empty dynamic protobuf and stats files: 19/19.
- Parseable enhanced JSON and retained SASS/CUBIN: 19/19.
- SASS files / CUBIN files: 38 / 38.
- Total dynamic launches: 676.
- Dataset apparent size before compression: 112,860,819,165 bytes
  (approximately 105.1 GiB).
- Archive: `A6000_compressed_dataset/lonestargpu-2.0.tar.zst`.
- Archive size: 12,380,222,127 bytes (approximately 11.5 GiB).
- SHA-256: `22c0c17b360c56e20e6e83fccb83b71fa84beb220de6370ad29174e31e135818`.
- Validation: the `.partial` archive passed complete `zstd -t` validation
  against a 113,608,785,920-byte tar stream before being renamed.
- Space policy: after archive validation, `A6000_dataset/lonestargpu-2.0`
  was deleted; its detailed `TRACE_MANIFEST.md` remains inside the archive.

## Proxy Apps DOE Final Validation and Archived Storage

- Script exit status after the cutil compatibility fix: success.
- Application directories / configured runs: 3 / 3.
- Non-empty dynamic protobuf and stats files: 3/3.
- Parseable enhanced JSON and retained SASS/CUBIN: 3/3.
- SASS files / CUBIN files: 5 / 5.
- Total dynamic launches: 38 (CNS 9, XSBench 1, LULESH 28).
- Dataset apparent trace size before adding the manifest: 948,736,063 bytes.
- Archive: `A6000_compressed_dataset/proxy-apps-doe.tar.zst`.
- Archive size: 17,214,007 bytes (approximately 16.4 MiB).
- SHA-256: `0243b9a5aa8141de53e34a10d8c5546ab265c93f8d96224add561ac7f32e66fb`.
- Validation: the `.partial` archive passed complete `zstd -t` validation
  against a 962,017,280-byte tar stream before being renamed.
- Space policy: after archive validation, `A6000_dataset/proxy-apps-doe` was
  deleted; its detailed `TRACE_MANIFEST.md` remains inside the archive.

## Dragon Naive Final Validation and Archived Storage

- Generation/recovery exit status after dependency and Join fixes: success.
- Application directories / configured runs: 4 / 8.
- Non-empty dynamic protobuf and stats files: 8/8.
- Parseable enhanced JSON and retained SASS/CUBIN: 8/8.
- SASS files / CUBIN files: 16 / 16.
- Total dynamic launches: 164.
- Dataset apparent trace size before adding the manifest: 10,132,867,545 bytes
  (approximately 9.44 GiB).
- Archive: `A6000_compressed_dataset/dragon-naive.tar.zst`.
- Archive size: 420,893,215 bytes (approximately 401 MiB).
- SHA-256: `2c91f687eca54ccc86c24d14c759934e59b7f10f926fd337938ba1906c61b52f`.
- Validation: the `.partial` archive passed complete `zstd -t` validation
  against a 10,185,902,080-byte tar stream before being renamed.
- Space policy: after archive validation, `A6000_dataset/dragon-naive` was
  deleted; its detailed `TRACE_MANIFEST.md` remains inside the archive.

## Dragon CDP Final Validation and Archived Storage

- Application directories / configured runs: 2 / 6.
- Tracer-visible parent/contract launches: 145; child CDP grids are represented
  within the parent execution rather than as independent host launch rows.
- Enhanced static kernels: 42; all completed runs retain SASS.
- Archive: `A6000_compressed_dataset/dragon-cdp.tar.zst`.
- Archive size: 407,037,113 bytes.
- SHA-256: `8b6eb9b30aa8715f79b7630619cf594482b5a71be09aa483518e7a5a983f8377`.
- Space policy: the validated archive is retained and the raw suite was
  deleted. The internal manifest records the CUDA 12.8 CDP metadata fix and
  the SSSP threshold adjustment used to avoid pending-launch saturation.

## GPU Microbenchmark Final Validation and Archived Storage

- Enabled applications / configured runs: 15 / 15.
- Dynamic launches / enhanced static kernels: 15 / 15.
- Non-empty SASS / CUBIN artifacts: 30 / 30.
- Archive: `A6000_compressed_dataset/GPU_Microbenchmark.tar.zst`.
- Archive size: 36,551,075 bytes.
- SHA-256: `5ed864ed1b6dc67ac9d7805ce13ded1262bd953ecff329d8b4a7a4179a77c627`.
- Space policy: the validated archive is retained and the raw suite was
  deleted. Traced bandwidth/latency console values include instrumentation
  overhead and are not native hardware performance measurements.

## DeepBench Tensor-Core GEMM Problems and Resolutions

- The aggregate NVIDIA DeepBench build also attempted convolution and failed
  on missing `cudnn.h`, although this trace suite enables only GEMM. Added a
  dedicated GEMM build target and changed its generation script to use it.
- Static cuBLAS linkage exposed hundreds of unrelated modules per run. The
  enhanced tracer now prefilters by dynamically observed symbols, parses only
  relevant functions, and removes unlaunched modules.
- A single run mixes local/CUDA helper `sm_86` kernels with `sm_80` cuBLAS
  kernels. The tracer formerly extracted only the architecture of the last
  launch; it now records and extracts every observed binary architecture and
  tags parsed SASS with the owning module architecture.
- Two CUTLASS kernels were JIT-generated and have no on-disk CUBIN/RFU. Their
  complete 3,960- and 1,616-instruction static maps were captured by NVBit,
  retained in enhanced JSON, and exported as explicitly marked standalone
  JIT SASS files. No CUBIN was fabricated.
- Several incomplete diagnostic attempts were interrupted and deleted before
  the final clean run. All nine final configurations completed normally.

## DeepBench Tensor-Core GEMM Final Validation and Archived Storage

- Application directories / configured runs: 1 / 9.
- Non-empty dynamic protobuf / stats / enhanced JSON: 9 / 9 / 9.
- Dynamic launches / enhanced static kernels: 78 / 48.
- Non-empty threadblock protobufs: 274,936.
- SASS / CUBIN / RFU artifacts: 67 / 65 / 65; the two-artifact difference is
  the documented JIT SASS with no on-disk binary.
- Dataset size before manifest: 15,419,472,878 bytes.
- Archive: `A6000_compressed_dataset/Deepbench_nvidia_tencore_gemm.tar.zst`.
- Archive size: 262,130,537 bytes.
- SHA-256: `e7d8647a74db04837ce8d447567a9310be476329312bc2ead55d872de1b67ed1`.
- Validation: `.partial` passed `zstd -t`; the complete tar listing contained
  275,349 members, the expected top-level path and manifest, and no member
  outside that prefix.
- Space policy: after validation and atomic rename, the raw DeepBench suite was
  deleted; its detailed `TRACE_MANIFEST.md` remains inside the archive.

## CUTLASS 5 Trace Problems and Resolutions

- The host initially lacked CMake. Installed CMake 3.28.3 and configured the
  CUTLASS build explicitly for the A6000's SM 8.6 architecture.
- The flattened source tree did not contain the GoogleTest submodule, although
  the performance binary included its header without using it. Made unit-test
  construction conditional on GoogleTest availability and removed the unused
  include; `cutlass_perf_test` then built successfully.
- Reduced every CUTLASS timing setting from 5 iterations to 1. The profiler
  still performs initial, warm-up, and timed launches of the same static
  kernel, so the CUTLASS script captures only launch 3. This keeps one complete
  representative dynamic trace and all static SASS/CUBIN/RFU artifacts.
- Interval capture exposed two tracer receiver bugs: skipped launches emitted
  flush tokens without captured kernel entries, and the receiver indexed a
  compact captured-kernel array using the global launch id. The receiver now
  consumes skipped flush tokens before binding a kernel and uses the latest
  captured entry for instruction data.
- CUTLASS's large template binary made old SASS/RFU parsing spend minutes on
  unlaunched functions. The tracer now skips irrelevant RFU bodies, guards
  annotation regexes, performs delimiter replacement in one linear pass, and
  retains only the launched module. A representative static post-process now
  completes in seconds while preserving the same parsed SASS.
- The original SGEMM `M=N=K=2560` workload was infeasible on the 56 GiB host.
  The smaller `N=1024,K=2560` run peaked around 27 GiB RSS; proportional block
  scaling predicts about 67 GiB for the original square run. Only its K was
  capped at 1024. The corrected `M=N=2560,K=1024` run kept all 400 output-grid
  blocks and the same SGEMM static kernel, peaked around 27.1 GiB, and completed.
- The old square run was interrupted after about three seconds and its 24 KiB
  partial directory was removed. The first 15 completed runs were retained;
  the corrected square and final four SGEMM runs completed through the recorded
  recovery subset.
- CUTLASS prints `NotVerified` because only its own provider is enabled and no
  cuBLAS reference provider runs. All 20 traced application processes printed
  `PASSED`, exited successfully, and passed artifact validation.

## CUTLASS 5 Trace Final Validation and Archived Storage

- Application directories / configured runs: 1 / 20.
- Program launches: 60 total (3 per run); captured timed launches: 20 total.
- Enhanced static kernels: 20; every kernel has non-empty SM 8.6 instructions.
- Threadblock protobuf files: 2,156.
- SASS / CUBIN / RFU artifacts: 20 / 20 / 20.
- Dataset size including manifest and generation log: 31,966,441,954 bytes.
- Archive: `A6000_compressed_dataset/cutlass_5_trace.tar.zst`.
- Archive size: 100,357,057 bytes (approximately 95.7 MiB), 0.3139% of the raw
  dataset size (approximately 318.5x smaller).
- SHA-256: `1e535990bb0de9e986cd04b66e6dc7d14ba133c3344a69a5aef89c4c0f888f41`.
- Validation: the `.partial` archive passed `zstd -t` against a complete
  31,970,775,040-byte tar stream. Its 2,540 members all use the expected
  `cutlass_5_trace/` prefix and contain the manifest plus exactly 20 run files,
  dynamic protobufs, enhanced JSON files, and SASS/CUBIN/RFU sets.
- Space policy: after validation and atomic rename, the raw CUTLASS suite was
  deleted; its `TRACE_MANIFEST.md` and console log remain inside the archive.
