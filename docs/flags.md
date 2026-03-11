# Configuration Flags

All flags are passed as JVM system properties (`-D`) unless noted otherwise.

---

## Measurement

| Flag | Default | Description |
|---|---|---|
| `-Dbenchmark.runs=N` | `100` | Number of timed iterations reported in the output table |
| `-Dbenchmark.warmup=N` | `20` | Number of warm-up iterations (excluded from timing) |

---

## Input sizes

Each benchmark has a named key. Override its default size with:

```
-Dbenchmark.<key>.size=N
```

| Key | Default size | Benchmark |
|---|---|---|
| `mxm` | 1024 | Matrix Multiplication (FP32) |
| `mxmfp16` | 1024 | Matrix Multiplication (FP16) |
| `mxv` | 16384 | Matrix-Vector Multiplication |
| `mt` | 8192 | Matrix Transpose |
| `dft` | 8192 | Discrete Fourier Transform |
| `montecarlo` | 134217728 | Montecarlo |
| `mandelbrot` | 512 | Mandelbrot |
| `nbody` | 16384 | N-Body |
| `juliaset` | 4096 | Julia Sets |
| `blackscholes` | 33554432 | Black-Scholes |
| `saxpy` | 67108864 | SAXPY |
| `rmsnorm` | 1024 | RMS Normalisation |
| `softmax` | 1024 | Softmax |
| `silu` | 1536 | SiLU |

Example — run matrix multiplication at 2048×2048:

```bash
./run.sh mxm -Dbenchmark.mxm.size=2048
```

---

## TornadoVM device selection

```
-Dbenchmark.device=<platform>:<device>
```

Default: `0:0` (first platform, first device).

List available devices:

```bash
tornado --listDevices
```

Example — run on device `0:1`:

```bash
./run.sh mxm -Dbenchmark.device=0:1
```

---

## Backend selection (positional argument)

Pass a mode keyword after the benchmark name to restrict which backends run:

| Argument | Backends executed |
|---|---|
| *(none)* | Sequential, Streams, Threads, VectorAPI, TornadoVM |
| `onlyJavaSeq` | Sequential only |
| `onlyJava` | Sequential, Streams, Threads, VectorAPI |
| `onlyTornadoVM` | Sequential (reference run) + TornadoVM |
| `jmh` | Run via JMH instead of the custom harness |

Example — TornadoVM only:

```bash
./run.sh softmax onlyTornadoVM
```

---

## Combining flags

```bash
./run.sh mxm onlyTornadoVM \
  -Dbenchmark.runs=20 \
  -Dbenchmark.warmup=5 \
  -Dbenchmark.mxm.size=2048 \
  -Dbenchmark.device=0:1
```
