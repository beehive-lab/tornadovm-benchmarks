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

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
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
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Benchmark taken from the TornadoVM internal suite.
 *
 * <p>
 *     How to run it?
 *     <code>
 *         java -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.Montecarlo
 *     </code>
 * </p>
 */
public class Montecarlo extends TornadoBenchmark {

    static final int SIZE = 16777216 * 8;

    private static void computeSequential(FloatArray output, int iterations) {
        for (@Parallel int j = 0; j < iterations; j++) {
            long seed = j;
            // generate a pseudo random number (you do need it twice)
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);

            // this generates a number between 0 and 1 (with an awful entropy)
            float x = (seed & 0x0FFFFFFF) / 268435455f;

            // repeat for y
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            float y = (seed & 0x0FFFFFFF) / 268435455f;

            float dist = (float) Math.sqrt(x * x + y * y);
            if (dist <= 1.0f) {
                output.set(j, 1.0f);
            } else {
                output.set(j, 0.0f);
            }
        }
    }

    private static void computeWithJavaStreams(FloatArray output, final int iterations) {
        IntStream.range(0, iterations).parallel().forEach(j -> {
            long seed = j;
            // generate a pseudo random number (you do need it twice)
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);

            // this generates a number between 0 and 1 (with an awful entropy)
            float x = (seed & 0x0FFFFFFF) / 268435455f;

            // repeat for y
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            float y = (seed & 0x0FFFFFFF) / 268435455f;

            float dist = (float) Math.sqrt(x * x + y * y);
            if (dist <= 1.0f) {
                output.set(j, 1.0f);
            } else {
                output.set(j, 0.0f);
            }
        });
    }

    private static void computeWithJavaThreads(FloatArray output) throws InterruptedException {

        Range[] ranges = Utils.createRangesForCPU(output.getSize());
        final int maxProcessors = Runtime.getRuntime().availableProcessors();

        Thread[] threads = new Thread[maxProcessors];
        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                for (int j = ranges[t].min(); j < ranges[t].max(); j++) {
                    long seed = j;
                    // generate a pseudo random number (you do need it twice)
                    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);

                    // this generates a number between 0 and 1 (with an awful entropy)
                    float x = (seed & 0x0FFFFFFF) / 268435455f;

                    // repeat for y
                    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                    float y = (seed & 0x0FFFFFFF) / 268435455f;

                    float dist = (float) Math.sqrt(x * x + y * y);
                    if (dist <= 1.0f) {
                        output.set(j, 1.0f);
                    } else {
                        output.set(j, 0.0f);
                    }
                }
            });
        });

        for (Thread t : threads) {
            t.start();
        }

        for (Thread t : threads) {
            t.join();
        }
    }

    private static void computeWithParallelVectorAPI(FloatArray output, final int iterations) {
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        for (int j = 0; j < iterations; j+= species.length()) {

            // Create vector x and vector y
            float[] x = new float[species.length()];
            float[] y = new float[species.length()];
            for (int i = 0; i < species.length(); i++) {
                long seed = j + i;
                // generate a pseudo random number (you do need it twice)
                seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);

                // this generates a number between 0 and 1 (with an awful entropy)
                x[i] = (seed & 0x0FFFFFFF) / 268435455f;

                // repeat for y
                seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                y[i] = (seed & 0x0FFFFFFF) / 268435455f;
            }

            FloatVector vX = FloatVector.fromArray(species, x, 0);
            FloatVector vY = FloatVector.fromArray(species, y, 0);
            FloatVector mulX = vX.mul(vX);
            FloatVector mulY = vY.mul(vY);
            float[] dist = mulX.add(mulY).sqrt().toArray();

            for (float v : dist) {
                if (v <= 1.0f) {
                    output.set(j, 1.0f);
                } else {
                    output.set(j, 0.0f);
                }
            }
        }
    }

    private static void computeWithTornadoVM(FloatArray output, final int iterations) {
        for (@Parallel int j = 0; j < iterations; j++) {
            long seed = j;
            // generate a pseudo random number (you do need it twice)
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);

            // this generates a number between 0 and 1 (with an awful entropy)
            float x = (seed & 0x0FFFFFFF) / 268435455f;

            // repeat for y
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            float y = (seed & 0x0FFFFFFF) / 268435455f;

            float dist = (float) Math.sqrt(x * x + y * y);
            if (dist <= 1.0f) {
                output.set(j, 1.0f);
            } else {
                output.set(j, 0.0f);
            }
        }
    }

    private boolean validate(FloatArray outputRef, FloatArray output) {
        for (int i = 0; i < outputRef.getSize(); i++) {
            if (Math.abs(outputRef.get(i) - output.get(i)) > 0.1) {
                System.out.println(outputRef.get(i) + " != " + output.get(i));
                return false;
            }
        }
        return true;
    }

    private void validate(int run, FloatArray outputRef, FloatArray output) {
        if (run == 0) {
            System.out.println(" -- Result Correct? " + validate(outputRef, output));
        } else {
            System.out.println();
        }
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {

        private Montecarlo montecarlo;

        private FloatArray output;

        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            final int size = SIZE;
            montecarlo = new Montecarlo();
            output = new FloatArray(size);
            TaskGraph taskGraph = new TaskGraph("benchmark")
                    .task("montecarlo", Montecarlo::computeWithTornadoVM, output, SIZE)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, output);
            executionPlan = new TornadoExecutionPlan(taskGraph.snapshot());
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void montecarloSequential(JMHBenchmark state) {
            computeSequential(state.output, SIZE);
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void montecarloParallelStreams(JMHBenchmark state) {
            state.montecarlo.computeWithJavaStreams(state.output, SIZE);
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void montecarloParallelThreads(JMHBenchmark state) throws InterruptedException {
            state.montecarlo.computeWithJavaThreads(state.output);
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void montecarloParallelVectorAPI(JMHBenchmark state) {
            state.montecarlo.computeWithParallelVectorAPI(state.output, SIZE);
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void montecarloTornadoVM(JMHBenchmark state) {
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
                .include(Montecarlo.class.getName() + ".*") //
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

        FloatArray outputSeq = new FloatArray(size);
        FloatArray outputStream = new FloatArray(size);
        FloatArray outputThreads = new FloatArray(size);
        FloatArray outputVector = new FloatArray(size);
        FloatArray outputTornadoVM = new FloatArray(size);


        // 5 implementations to compare
        final int implementationsToCompare = 5;
        ArrayList<ArrayList<Long>> timers = IntStream.range(0, implementationsToCompare) //
                .<ArrayList<Long>>mapToObj(i -> new ArrayList<>()) //
                .collect(Collectors.toCollection(ArrayList::new));

        for (int i = 0; i < Config.RUNS; i++) {
            long start = System.nanoTime();
            computeSequential(outputSeq, SIZE);
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
                computeWithJavaStreams(outputStream, SIZE);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(1).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Stream Elapsed time: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                validate(i, outputSeq, outputStream);
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
                validate(i, outputSeq, outputThreads);
            }

            // 4. Parallel with Java Vector API
            for (int i = 0; i < Config.RUNS; i++) {
                long start = System.nanoTime();
                computeWithParallelVectorAPI(outputVector, SIZE);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(3).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Elapsed time Parallel Vectorized: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                validate(i, outputSeq, outputVector);
            }
        }

        if (option == Option.ALL || option == Option.TORNADO_ONLY) {
            // TornadoVM
            TaskGraph taskGraph = new TaskGraph("benchmark")
                    .task("montecarlo", Montecarlo::computeWithTornadoVM, outputTornadoVM, SIZE)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outputTornadoVM);
            try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot())) {

                TornadoDevice device = TornadoExecutionPlan.getDevice(0, 0);
                executionPlan.withDevice(device);

                // 5. On the GPU using TornadoVM
                for (int i = 0; i < Config.RUNS; i++) {
                    long start = System.nanoTime();
                    executionPlan.execute();
                    long end = System.nanoTime();
                    long elapsedTime = (end - start);
                    timers.get(4).add(elapsedTime);
                    double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                    System.out.print("Elapsed time TornadoVM-GPU: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                    validate(i, outputSeq, outputTornadoVM);
                }
            } catch (TornadoExecutionPlanException e) {
                throw new RuntimeException(e);
            }
        }

        if (option == Option.ALL) {
            Utils.dumpPerformanceTable(timers, implementationsToCompare, "montecarlo", Config.HEADER1);
        }
    }

    @Override
    String getName() {
        return "Montecarlo";
    }

    @Override
    String printSize() {
        return "" + getSize();
    }

    public static void main(String[] args) throws InterruptedException {
        Montecarlo benchmark = new Montecarlo();
        benchmark.run(args);
    }
}
