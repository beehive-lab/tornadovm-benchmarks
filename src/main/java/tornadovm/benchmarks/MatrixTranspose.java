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
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DFloat;

import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * How to run?
 *
 * <code>
 *     tornado -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.MatrixTranspose
 * </code>
 */
public class MatrixTranspose extends Benchmark {

    final static int SIZE = 8192;

    public static void computeSequential(Matrix2DFloat a, Matrix2DFloat b) {
        for (int i = 0; i < a.getNumRows(); i++) {
            for (int j = 0; j < b.getNumColumns(); j++) {
                b.set(j, i, a.get(i, j));
            }
        }
    }

    public static void computeWithJavaStreams(Matrix2DFloat a, Matrix2DFloat b) {
        IntStream.range(0, a.getNumRows()).parallel().forEach(i -> {
            IntStream.range(0, b.getNumColumns()).parallel().forEach(j -> {
                b.set(j, i, a.get(i, j));
            });
        });
    }

    public static void computeWithJavaThreads(Matrix2DFloat a, Matrix2DFloat b) throws InterruptedException {
        Range[] ranges = Utils.createRangesForCPU(SIZE);
        int maxProcessors = Runtime.getRuntime().availableProcessors();
        Thread[] threads = new Thread[maxProcessors];

        IntStream.range(0, threads.length).forEach(threadIndex -> {
            threads[threadIndex] = new Thread(() -> {
                for (int i = ranges[threadIndex].min(); i < ranges[threadIndex].max(); i++) {
                    for (int j = 0; j < b.getNumColumns(); j++) {
                        b.set(j, i, a.get(i, j));
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

    // TODO: not sure if it is possible to compute this algorithm efficiently using the vector API
    public static void computeWithParallelVectorAPI(Matrix2DFloat a, Matrix2DFloat b) {
        VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
        final int vectorLength = SPECIES.length();
        IntStream.range(0, a.getNumRows()).parallel().forEach(i -> {
            for (int j = 0; j < b.getNumColumns(); j += vectorLength) {
                FloatVector vector = FloatVector.fromMemorySegment(SPECIES, a.getSegment(), ((i * a.getNumRows()) + j) * 4, ByteOrder.nativeOrder());
                for (int k = 0; k < vectorLength; k++) {
                    if (j + k < b.getNumColumns()) {
                        b.set(j + k, i, vector.lane(k));
                    }
                }
            }
        });
    }

    public static void computeWithTornadoVM(Matrix2DFloat a, Matrix2DFloat b) {
        for (@Parallel int i = 0; i < a.getNumRows(); i++) {
            for (@Parallel int j = 0; j < b.getNumColumns(); j++) {
                b.set(j, i, a.get(i, j));
            }
        }
    }

    private static boolean verify(Matrix2DFloat a, Matrix2DFloat b) {
        for (int i = 0; i < a.getNumRows(); i++) {
            for (int j = 0; j < a.getNumColumns(); j++) {
                if (Math.abs(a.get(i, j) - b.get(i, j)) > Config.DELTA) {
                    return false;
                }
            }
        }
        return true;
    }

    private static void validate(int run, Matrix2DFloat matrix, Matrix2DFloat reference) {
        System.out.println(" -- Result Correct? " + verify(matrix, reference));
//        if (run == 0) {
//            System.out.println(" -- Result Correct? " + verify(matrix, reference));
//        } else {
//            System.out.println();
//        }
    }

    @Override
    int getSize() {
        return SIZE;
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {

        private MatrixTranspose matrixTranspose;
        private Matrix2DFloat matrix;
        private Matrix2DFloat output;
        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            matrixTranspose = new MatrixTranspose();
            matrix = new Matrix2DFloat(SIZE, SIZE);
            output = new Matrix2DFloat(SIZE, SIZE);
            TaskGraph taskGraph = new TaskGraph("benchmark")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrix)
                    .task("matrixTranspose", MatrixTranspose::computeWithTornadoVM, matrix, output)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

            executionPlan = new TornadoExecutionPlan(taskGraph.snapshot());
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void matrixTransposeSequential(JMHBenchmark state) {
            computeSequential(state.matrix, state.output);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void matrixTransposeParallelStreams(JMHBenchmark state) {
            state.matrixTranspose.computeWithJavaStreams(state.matrix, state.output);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void matrixTransposeParallelThreads(JMHBenchmark state) throws InterruptedException {
            state.matrixTranspose.computeWithJavaThreads(state.matrix, state.output);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void matrixTransposeParallelVectorAPI(JMHBenchmark state) {
            state.matrixTranspose.computeWithParallelVectorAPI(state.matrix, state.output);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void matrixTransposeTornadoVM(JMHBenchmark state) {
            state.executionPlan.execute();
        }

    }

    @Override
    void runWithJMH() throws RunnerException {
        org.openjdk.jmh.runner.options.Options opt = new OptionsBuilder() //
                .include(MatrixTranspose.class.getName() + ".*") //
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
    void runTestAll(int size, Option option) throws InterruptedException {

        // Using Panama Segments
        Matrix2DFloat matrixA = new Matrix2DFloat(size, size);

        // Matrix for results
        Matrix2DFloat outputReference = new Matrix2DFloat(size, size);
        Matrix2DFloat outputStreams = new Matrix2DFloat(size, size);
        Matrix2DFloat outputThreads = new Matrix2DFloat(size, size);
        Matrix2DFloat outputVector = new Matrix2DFloat(size, size);
        Matrix2DFloat outputTornado = new Matrix2DFloat(size, size);


        Random r = new Random(71);
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                matrixA.set(i, j, r.nextFloat());
            }
        }

        final int implementationsToCompare = 5;
        // 6 implementations to compare
        ArrayList<ArrayList<Long>> timers = IntStream.range(0, implementationsToCompare) //
                .<ArrayList<Long>>mapToObj(i -> new ArrayList<>()) //
                .collect(Collectors.toCollection(ArrayList::new));

        // 1. Sequential
        for (int i = 0; i < Config.RUNS; i++) {
            long start = System.nanoTime();
            computeSequential(matrixA, outputReference);
            long end = System.nanoTime();
            long elapsedTime = (end - start);
            timers.get(0).add(elapsedTime);
            double elapsedTimeMilliseconds = elapsedTime * 1E-6;

            System.out.println("Elapsed time: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) ");

            if (option == Option.TORNADO_ONLY) {
                // We only run one iteration just to run the reference implementation to check results.
                break;
            }
        }

        if (option == Option.ALL || option == Option.JAVA_ONLY) {
            // 2. Parallel Streams
            for (int i = 0; i < Config.RUNS; i++) {
                long start = System.nanoTime();
                computeWithJavaStreams(matrixA, outputStreams);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(1).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Stream Elapsed time: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                validate(i, outputStreams, outputReference);
            }

            // 3. Parallel with Java Threads
            for (int i = 0; i < Config.RUNS; i++) {
                long start = System.nanoTime();
                computeWithJavaThreads(matrixA, outputThreads);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(2).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Elapsed time Threads: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                validate(i, outputThreads, outputReference);
            }

            // 4. Parallel with Java Vector API
            for (int i = 0; i < Config.RUNS; i++) {
                try {
                    long start = System.nanoTime();
                    computeWithParallelVectorAPI(matrixA, outputVector);
                    long end = System.nanoTime();
                    long elapsedTime = (end - start);
                    timers.get(3).add(elapsedTime);
                    double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                    System.out.print("Elapsed time Parallel Vectorized: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                    validate(i, outputVector, outputReference);
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
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, matrixA)
                    .task("matrixTranspose", MatrixTranspose::computeWithTornadoVM, matrixA, outputTornado)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outputTornado);

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
                    validate(i, outputTornado, outputReference);
                }
            } catch (TornadoExecutionPlanException e) {
                throw new RuntimeException(e);
            }
        }

        if (option == Option.ALL) {
            Utils.dumpPerformanceTable(timers, implementationsToCompare, "matrixTranspose", Config.HEADER1);
        }
    }

    @Override
    String getName() {
        return "MatrixTranspose";
    }

    @Override
    String printSize() {
        return getSize() + "x" + getSize();
    }
}
