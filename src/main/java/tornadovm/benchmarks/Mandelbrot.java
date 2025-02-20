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

import org.openjdk.jmh.annotations.Benchmark;
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
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * <code>
 *     tornado -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.Mandelbrot
 * </code>
 */
public class Mandelbrot extends TornadoBenchmark {
    
    static final int SIZE = 512;
    static final int ITERATIONS = 10000;

    private ShortArray computeSequential(int size) {
        final int iterations = ITERATIONS;
        float space = 2.0f / size;

        ShortArray result = new ShortArray(size * size);

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
                result.set(i * size + j, r);
            }
        }
        return result;
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

    private void computeWithJavaStreams(int size, ShortArray output) {
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

    private void computeWithJavaThreads(ShortArray output) throws InterruptedException {
        Range[] ranges = Utils.createRangesForCPU(SIZE);
        final int iterations = ITERATIONS;
        float space = 2.0f / Mandelbrot.SIZE;

        int maxProcessors = Runtime.getRuntime().availableProcessors();
        Thread[] threads = new Thread[maxProcessors];
        IntStream.range(0, threads.length).forEach(threadIndex -> {
            threads[threadIndex] = new Thread(() -> {
                for (int i = ranges[threadIndex].min(); i < ranges[threadIndex].max(); i++) {
                    for (int j = 0; j < Mandelbrot.SIZE; j++) {
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
                        output.set(i * Mandelbrot.SIZE + j, r);
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

    @State(Scope.Thread)
    public static class JMHBenchmark {
        private Mandelbrot benchmark;
        private ShortArray output;
        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            benchmark = new Mandelbrot();
            output = new ShortArray(SIZE * SIZE);
            TaskGraph taskGraph = new TaskGraph("benchmark")
                    .task("mandelbrot", Mandelbrot::computeWithTornadoVM, SIZE, output)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, output);
            executionPlan = new TornadoExecutionPlan(taskGraph.snapshot());
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mandelbrotSequential(JMHBenchmark state) {
            state.benchmark.computeSequential(SIZE);
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mandelbrotParallelStreams(JMHBenchmark state) {
            state.benchmark.computeWithJavaStreams(SIZE,  state.output);
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mandelbrotParallelThreads(JMHBenchmark state) throws InterruptedException {
            state.benchmark.computeWithJavaThreads(state.output);
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mandelbrotParallelVectorAPI(JMHBenchmark state) {
            state.benchmark.computeWithParallelVectorAPI(SIZE, state.output);
        }

        @Benchmark
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
        return SIZE;
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
    void runTestAll(final int size, Option option) throws InterruptedException {

        ShortArray outputSeq = new ShortArray(size * size);
        ShortArray outputStream = new ShortArray(size * size);
        ShortArray outputThreads = new ShortArray(size * size);
        ShortArray outputVector = new ShortArray(size * size);
        ShortArray outputTornadoVM = new ShortArray(size * size);

        // 5 implementations to compare
        final int implementationsToCompare = 5;
        ArrayList<ArrayList<Long>> timers = IntStream.range(0, implementationsToCompare) //
                .<ArrayList<Long>>mapToObj(i -> new ArrayList<>()) //
                .collect(Collectors.toCollection(ArrayList::new));

        for (int i = 0; i < Config.RUNS; i++) {
            long start = System.nanoTime();
            outputSeq = computeSequential(SIZE);
            long end = System.nanoTime();
            long elapsedTime = (end - start);
            timers.get(0).add(elapsedTime);
            double elapsedTimeMilliseconds = elapsedTime * 1E-6;

            System.out.println("Elapsed time: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");

            if (option == Option.TORNADO_ONLY) {
                // We only run one iteration just to run the reference implementation to check results.
                break;
            }
        }

        if (option == Option.ALL || option == Option.JAVA_ONLY) {
            // 2. Parallel Streams
            for (int i = 0; i < Config.RUNS; i++) {
                long start = System.nanoTime();
                computeWithJavaStreams(SIZE, outputStream);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(1).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Stream Elapsed time: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                if (i == 0) {
                    System.out.println(" -- Result Correct? " + validate(outputSeq, outputStream, size));
                } else {
                    System.out.println();
                }
            }

            // 3. Parallel with Java Threads
            for (int i = 0; i < Config.RUNS; i++) {
                long start = System.nanoTime();
                computeWithJavaThreads(outputThreads);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(2).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Elapsed time Threads: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");

                if (i == 0) {
                    System.out.println(" -- Result Correct? " + validate(outputSeq, outputThreads, size));
                } else {
                    System.out.println();
                }
            }

            // 4. Parallel with Java Vector API
            for (int i = 0; i < Config.RUNS; i++) {
                long start = System.nanoTime();
                try {
                    computeWithParallelVectorAPI(SIZE, outputVector);
                    long end = System.nanoTime();
                    long elapsedTime = (end - start);
                    timers.get(3).add(elapsedTime);
                    double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                    System.out.print("Elapsed time Parallel Vectorized: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                    if (i == 0) {
                        System.out.println(" -- Result Correct? " + validate(outputSeq, outputVector, size));
                    } else {
                        System.out.println();
                    }

                } catch (RuntimeException e) {
                    System.out.println("Error - Parallel Vector API: " + e.getMessage());
                    // We store -1 in the timers list to indicate that an error has occurred.
                    timers.get(3).add((long) -1);
                }
            }
        }

        if (option == Option.ALL || option == Option.TORNADO_ONLY) {
            // TornadoVM
            TaskGraph taskGraph = new TaskGraph("benchmark")
                    .task("mandelbrot", Mandelbrot::computeWithTornadoVM, SIZE, outputTornadoVM)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outputTornadoVM);
            try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                // 5. On the GPU using TornadoVM
                for (int i = 0; i < Config.RUNS; i++) {
                    long start = System.nanoTime();
                    executionPlan.execute();
                    long end = System.nanoTime();
                    long elapsedTime = (end - start);
                    timers.get(4).add(elapsedTime);
                    double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                    System.out.print("Elapsed time TornadoVM-GPU: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                    if (i == 0) {
                        System.out.println(" -- Result Correct? " + validate(outputSeq, outputTornadoVM, size));
                    } else {
                        System.out.println();
                    }
                }
            } catch (TornadoExecutionPlanException e) {
                throw new RuntimeException(e);
            }
        }

        if (option == Option.ALL) {
            Utils.dumpPerformanceTable(timers, implementationsToCompare, "mandelbrot", Config.HEADER1);
        }
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
        Mandelbrot benchmark = new Mandelbrot();
        benchmark.run(args);
    }
}
