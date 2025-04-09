# TornadoVM Benchmark Suite

TornadoVM Benchmark Suite. **This is a work in progress** and it is a framework to  compare 
the TornadoVM applications with Java Streams and Java Vector API. 
Not all implementations contain the Java Vector API at the moment. 


Note: this benchmarking suite is currently under development and definition. 
Some kernels may not be suitable due to lack of relevance or input size limitations 
on certain accelerators. The suite aims to showcase code diversification, with a focus on 
LLM, physics, and math simulation workloads.


## How to build?


```bash
./build.sh
```

Then install TornadoVM in a separated directory:

```bash
git clone https://github.com/beehive-lab/TornadoVM
cd tornadovm 
./bin/tornadovm-installer --backend=opencl --jdk jdk21 
cp setvars.sh .. 
cd ..
```

## How to run? 

Setup the environment:

```bash
source setvars.sh
```

### Run Individual benchamrk:

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

## Run all:

```bash
./run.sh 
```

## Run with JMH 

```bash
./run.sh <benchmark> jmh
```

For example, to run `mxm` with `jmh`:

```bash
./run.sh mxm jmh
```

## How to Change Device for an Specific Benchmark? 

For example, device `0:2` for the benchmark `mxv`:

```bash
tornado --printKernel --jvm="-Dtornado.device.memory=2GB -Dbenchmark.mxv.device=0:2" -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.Main mxv
```
    
