/*
 * Copyright (c) 2025, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package tornadovm.benchmarks;

import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;

import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * <code>
 *     tornado -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.Mandelbrot
 * </code>
 */
public class Mandelbrot extends BenchmarkDriver {

    public int size;
    static final int ITERATIONS = 10000;
    ShortArray resultSeq;
    ShortArray output;

    public Mandelbrot(int size) {
        this.size = size;
        resultSeq = new ShortArray(size * size);
        output = new ShortArray(size * size);
    }

    @Override
    public void computeSequential() {
        final int iterations = ITERATIONS;
        float space = 2.0f / size;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float Zr = 0.0f;
                float Zi = 0.0f;
                float Cr = (1 * j * space - 1.5f);
                float Ci = (1 * i * space - 1.0f);
                float ZrN = 0;
                float ZiN = 0;
                int y;
                for (y = 0; y < iterations && ZiN + ZrN <= 4.0f; y++) {
                    Zi = 2.0f * Zr * Zi + Ci;
                    Zr = 1 * ZrN - ZiN + Cr;
                    ZiN = Zi * Zi;
                    ZrN = Zr * Zr;
                }
                short r = (short) ((y * 255) / iterations);
                resultSeq.set(i * size + j, r);
            }
        }
    }

    private static void computeWithTornadoVM(int size, ShortArray output) {
        final int iterations = ITERATIONS;
        float space = 2.0f / size;
        for (@Parallel int i = 0; i < size; i++) {
            for (@Parallel int j = 0; j < size; j++) {
                float Zr = 0.0f;
                float Zi = 0.0f;
                float Cr = (1 * j * space - 1.5f);
                float Ci = (1 * i * space - 1.0f);
                float ZrN = 0;
                float ZiN = 0;
                int y = 0;
                for (int ii = 0; ii < iterations; ii++) {
                    if (ZiN + ZrN <= 4.0f) {
                        Zi = 2.0f * Zr * Zi + Ci;
                        Zr = 1 * ZrN - ZiN + Cr;
                        ZiN = Zi * Zi;
                        ZrN = Zr * Zr;
                        y++;
                    } else {
                        ii = iterations;
                    }
                }
                short r = (short) ((y * 255) / iterations);
                output.set(i * size + j, r);
            }
        }
    }

    @Override
    public void computeWithJavaStreams() {
        final int iterations = ITERATIONS;
        float space = 2.0f / size;
        IntStream.range(0, size).parallel().forEach(i -> {
            IntStream.range(0, size).parallel().forEach(j -> {
                float Zr = 0.0f;
                float Zi = 0.0f;
                float Cr = (1 * j * space - 1.5f);
                float Ci = (1 * i * space - 1.0f);
                float ZrN = 0;
                float ZiN = 0;
                int y = 0;
                for (int ii = 0; ii < iterations; ii++) {
                    if (ZiN + ZrN <= 4.0f) {
                        Zi = 2.0f * Zr * Zi + Ci;
                        Zr = 1 * ZrN - ZiN + Cr;
                        ZiN = Zi * Zi;
                        ZrN = Zr * Zr;
                        y++;
                    } else {
                        ii = iterations;
                    }
                }
                short r = (short) ((y * 255) / iterations);
                output.set(i * size + j, r);
            });
        });
    }

    @Override
    public void computeWithJavaThreads() throws InterruptedException {
        Range[] ranges = Utils.createRangesForCPU(size);
        final int iterations = ITERATIONS;
        float space = 2.0f / size;

        int maxProcessors = Runtime.getRuntime().availableProcessors();
        Thread[] threads = new Thread[maxProcessors];
        IntStream.range(0, threads.length).forEach(threadIndex -> {
            threads[threadIndex] = new Thread(() -> {
                for (int i = ranges[threadIndex].min(); i < ranges[threadIndex].max(); i++) {
                    for (int j = 0; j < size; j++) {
                        float Zr = 0.0f;
                        float Zi = 0.0f;
                        float Cr = (1 * j * space - 1.5f);
                        float Ci = (1 * i * space - 1.0f);
                        float ZrN = 0;
                        float ZiN = 0;
                        int y = 0;
                        for (int ii = 0; ii < iterations; ii++) {
                            if (ZiN + ZrN <= 4.0f) {
                                Zi = 2.0f * Zr * Zi + Ci;
                                Zr = 1 * ZrN - ZiN + Cr;
                                ZiN = Zi * Zi;
                                ZrN = Zr * Zr;
                                y++;
                            } else {
                                ii = iterations;
                            }
                        }
                        short r = (short) ((y * 255) / iterations);
                        output.set(i * size + j, r);
                    }
                }
            });
        });
        for (Thread thread : threads) {
            thread.start();
        }
        for (Thread thread : threads) {
            thread.join();
        }
    }

    @Override
    public void computeWithParallelVectorAPI() {
        throw new RuntimeException("Not implemented yet");
    }

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        TaskGraph taskGraph = new TaskGraph("benchmark")
                .task("mandelbrot", Mandelbrot::computeWithTornadoVM, size, output)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);
        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    @Override
    public void resetOutputs() {
        output.init((short) 0);
    }

    @Override
    public void validate(int i) {
        validate(i, resultSeq, output, size);
    }

    // TODO: Implement with Vector API
    private void computeWithParallelVectorAPI(int size, ShortArray output) {
        throw new RuntimeException("Not implemented yet");
    }

    private boolean validate(ShortArray outputRef, ShortArray output, int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (Math.abs(outputRef.get(i * size + j) - output.get(i * size + j)) > 0.0f) {
                    System.out.println(outputRef.get(i * size + j) + " != " + output.get(i * size + j));
                    return false;
                }
            }
        }
        return true;
    }

    private void validate(int run, ShortArray outputRef, ShortArray output, int size) {
        if (run == 0) {
            System.out.println(" -- Result Correct? " + validate(outputRef, output, size));
        } else {
            System.out.println();
        }
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {
        private Mandelbrot benchmark;
        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            benchmark = new Mandelbrot(Catalog.DEFAULT.get(Catalog.BenchmarkID.Mandelbrot).size());
            executionPlan = benchmark.buildExecutionPlan();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mandelbrotSequential(JMHBenchmark state) {
            state.benchmark.computeSequential();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mandelbrotParallelStreams(JMHBenchmark state) {
            state.benchmark.computeWithJavaStreams();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mandelbrotParallelThreads(JMHBenchmark state) throws InterruptedException {
            state.benchmark.computeWithJavaThreads();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mandelbrotParallelVectorAPI(JMHBenchmark state) {
            state.benchmark.computeWithParallelVectorAPI();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mandelbrotTornadoVM(JMHBenchmark state) {
            state.executionPlan.execute();
        }
    }

    @Override
    int getSize() {
        return size;
    }

    @Override
    void runWithJMH() throws RunnerException {
        org.openjdk.jmh.runner.options.Options opt = new OptionsBuilder() //
                .include(Mandelbrot.class.getName() + ".*") //
                .mode(Mode.AverageTime) //
                .timeUnit(TimeUnit.NANOSECONDS) //
                .warmupTime(TimeValue.seconds(60)) //
                .warmupIterations(2) //
                .measurementTime(TimeValue.seconds(30)) //
                .measurementIterations(5) //
                .forks(1) //
                .build();
        new Runner(opt).run();
    }

    @Override
    String getName() {
        return "Mandelbrot";
    }

    @Override
    String printSize() {
        return getSize() + "x" + getSize();
    }

    public static void main(String[] args) throws InterruptedException {
        Mandelbrot benchmark = new Mandelbrot(Catalog.DEFAULT.get(Catalog.BenchmarkID.Mandelbrot).size());
        benchmark.run(args);
    }
}
