# TornadoVM Benchmark Suite

TornadoVM Benchmark Suite. **This is a work in progress** and it is a framework to  compare 
the TornadoVM applications with Java Streams and Java Vector API. 
Not all implementations contain the Java Vector API at the moment. 


Note: this benchmarking suite is currently under development and definition. 
Some kernels may not be suitable due to lack of relevance or input size limitations 
on certain accelerators. The suite aims to showcase code diversification, with a focus on 
LLM, physics, and math simulation workloads.


## Install the TornadoVM SDK on Linux or macOS

Ensure that your JAVA_HOME points to a supported JDK before using the SDK. Download an SDK package matching your OS, architecture, and accelerator backend (opencl, ptx).
TornadoVM is distributed through our [**official website**](https://www.tornadovm.org/downloads) and **SDKMAN!**. Install a version that matches your OS, architecture, and accelerator backend.

All TornadoVM SDKs are available on the [SDKMAN! TornadoVM page](https://sdkman.io/sdks/tornadovm/).

### SDKMAN! Installation (Recommended)

#### Install SDKMAN! if not installed already
```bash
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk version
```
#### Install TornadoVM via SDKMAN!
```bash
sdk install tornadovm
```

## Build the TornadoVM Benchmark Suite
```bash
mvn -Dstyle.color=always clean install
```

## Run the functionality tests of the TornadoVM Benchmark Suite
```bash
mvn test
```

## How to execute the TornadoVM Benchmark Suite?

### Run an individual benchmark:

```bash
# Matrix Multiplication
./run.sh mxm

# Matrix Vector
./run.sh mxv

# Mandelbrot
./run.sh mandelbrot

# Montecarlo
./run.sh motecarlo

# Run DFT
./run.sh dft

# Matrix Transpose
./run.sh mt
```

### Run all benchmarks:

```bash
./run.sh
```

### Run with a specific backend mode:

```bash
# Parallel Java backends only (Streams, Threads, VectorAPI)
./run.sh mxm onlyJavaPar

# Sequential Java only
./run.sh mxm onlyJavaSeq

# All Java backends (no TornadoVM)
./run.sh mxm onlyJava

# TornadoVM only
./run.sh mxm onlyTornadoVM
```

Mode keywords also work without a benchmark name to run all benchmarks:

```bash
./run.sh onlyJavaPar
```

### Validate results:

Pass `validate` to check each backend's output against the sequential reference:

```bash
./run.sh mxm onlyJavaPar validate
./run.sh onlyTornadoVM validate
```

### Run with JMH

```bash
./run.sh <benchmark> jmh
```

For example, to run `mxm` with `jmh`:

```bash
./run.sh mxm jmh
```

### How to Change Device for an Specific Benchmark?

For example, device `0:2` for the benchmark `mxv`:

```bash
tornado --printKernel --jvm="-Dtornado.device.memory=2GB -Dbenchmark.mxv.device=0:2" -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.Main mxv
```

### Configuration flags

For the full list of flags (input sizes, iteration counts, device selection, backend filtering):

see [docs/flags.md](docs/flags.md)

## Acknowledgments

This work has been supported by the following EU & UKRI grants (most recent first):

- EU Horizon Europe & UKRI [AERO 101092850](https://aero-project.eu/).
- EU Horizon Europe & UKRI [P2CODE 101093069](https://p2code-project.eu/).
- EU Horizon Europe & UKRI [ENCRYPT 101070670](https://encrypt-project.eu).
- EU Horizon Europe & UKRI [TANGO 101070052](https://tango-project.eu).