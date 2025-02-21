# TornadoVM Benchmarks 

TornadoVM Benchmark Suite.

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
./run.sh mxm

./run.sh mxv

./run.sh mandelbrot

./run.sh mandelbrot

./run.sh dft
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
    