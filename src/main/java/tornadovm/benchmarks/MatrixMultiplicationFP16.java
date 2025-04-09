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
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
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
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * How to run?
 * <code>
 *     tornado -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.MatrixMultiplicationFP16
 * </code>
 */
public class MatrixMultiplicationFP16 extends Benchmark {

    int size;
    private double FLOP;
    private final static float TIME_SCALE_SECS = 1.0E09f;

    public MatrixMultiplicationFP16(int size) {
        this.size = size;
        FLOP = 2 * Math.pow(size, 3);
    }

    /**
     * Float MxN Matrix
     */
    private static class FP16Matrix {

        private static final int HALF_FLOAT_SIZE = 2;

        private final int m;
        private final int n;
        private final MemorySegment segment;

        public FP16Matrix(int m, int n) {
            this.m = m;
            this.n = n;
            final long segmentByteSize = n * m * HALF_FLOAT_SIZE;
            segment = Arena.ofAuto().allocate(segmentByteSize, 64);
        }

        public void set(int i, int j, float value) {
            final int index = i * m + j;
            short val = Float.floatToFloat16(value);
            segment.set(JAVA_SHORT, index * HALF_FLOAT_SIZE, val);
        }

        public float get(int i, int j) {
            final int index = i * m + j;
            short val = segment.get(JAVA_SHORT, index * HALF_FLOAT_SIZE);
            return Float.float16ToFloat(val);
        }

        public void initRandom() {
            Random r = new Random(71);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    float val = r.nextFloat();
                    set(i, j, Float.floatToFloat16(val));
                }
            }
        }

        public int M() {
            return m;
        }

        public int N() {
            return n;
        }
    }

    private static class Multiplication {

        private static final boolean DEBUG = false;

        /**
         * Matrix Multiplication using Panama Segments Sequentially
         *
         * @param a
         * @param b
         * @param c
         */
        public static void mxmSequential(FP16Matrix a, FP16Matrix b, FP16Matrix c) {
            for (int i = 0; i < a.M(); i++) {
                for (int j = 0; j < b.N(); j++) {
                    float acc = 0;
                    for (int k = 0; k < c.M(); k++) {
                        acc += a.get(i, k) * b.get(k, j);
                    }
                    c.set(i, j, acc);
                }
            }
        }

        public static void mxmParallelStreams(FP16Matrix a, FP16Matrix b, FP16Matrix c) {
            IntStream.range(0, a.M()).parallel().forEach(i -> IntStream.range(0, b.N()).parallel().forEach(j -> {
                float acc = 0;
                for (int k = 0; k < c.M(); k++) {
                    acc += a.get(i, k) * b.get(k, j);
                }
                c.set(i, j, acc);
            }));
        }

        public static void mxmParallelThreads(FP16Matrix a, FP16Matrix b, FP16Matrix c) throws InterruptedException {

            int maxProcessors = Runtime.getRuntime().availableProcessors();
            Range[] ranges = Utils.createRangesForCPU(a.M());

            Thread[] threads = new Thread[maxProcessors];
            IntStream.range(0, threads.length).forEach(t -> {
                threads[t] = new Thread(() -> {
                    for (int i = ranges[t].min(); i < ranges[t].max(); i++) {
                        for (int j = 0; j < b.N(); j++) {
                            float acc = 0;
                            for (int k = 0; k < c.M(); k++) {
                                acc += a.get(i, k) * b.get(k, j);
                            }
                            c.set(i, j, acc);
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

        public static FP16Matrix transposeMatrix(FP16Matrix matrix) {
            FP16Matrix matrixTranspose = new FP16Matrix(matrix.M(), matrix.N());
            for (int i = 0; i < matrix.M(); i++) {
                for (int j = 0; j < matrix.N(); j++) {
                    matrixTranspose.set(i, j, matrix.get(j, i));
                }
            }
            return matrixTranspose;
        }


        public static void mxmSequentialVectorized(FP16Matrix a, FP16Matrix b, FP16Matrix c) {
            throw new UnsupportedOperationException("Vector API FP16 Not Supported");
        }

        public static void mxmParallelVectorized(FP16Matrix a, FP16Matrix b, FP16Matrix c) {
            throw new UnsupportedOperationException("Vector API FP16 Not Supported");
        }

        /**
         * This method computes squared matrix multiplication.
         * @param a
         * @param b
         * @param c
         * @param size (num rows and num columns)
         */
        private static void mxmTornadoVM(HalfFloatArray a, HalfFloatArray b, HalfFloatArray c, final int size) {
            for (@Parallel int i = 0; i < size; i++) {
                for (@Parallel int j = 0; j < size; j++) {
                    HalfFloat sum = new HalfFloat(0.0f);
                    for (int k = 0; k < size; k++) {
                        HalfFloat f1 = a.get(i * size +  k);
                        HalfFloat f2 = b.get(k * size +  j);
                        HalfFloat result = HalfFloat.mult(f1, f2);
                        sum = HalfFloat.add(sum, result);
                    }
                    c.set(i * size + j, sum);
                }
            }
        }

        public static HalfFloatArray transformMatrixForTornadoVM(FP16Matrix a) {
            final int m = a.M();
            final int n = a.N();
            final HalfFloatArray matrix = new HalfFloatArray(m * n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    matrix.set(i * m +  j, new HalfFloat(a.get(i, j)));
                }
            }
            return matrix;
        }

        private static TaskGraph createTaskGraph(HalfFloatArray a, HalfFloatArray b, HalfFloatArray c, int size) {
            TaskGraph taskGraph = new TaskGraph("benchmark");
            taskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                    .task("mxmfp16", Multiplication::mxmTornadoVM, a, b, c, size) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, c);
            return taskGraph;
        }

        private static boolean verify(FP16Matrix matrix, FP16Matrix referenceMatrix) {
            boolean check = true;
            for (int i = 0; i < matrix.M(); i++) {
                for (int j = 0; j < matrix.N(); j++) {
                    if (Math.abs(matrix.get(i, j) - referenceMatrix.get(i, j)) > 0.1f) {
                        System.out.println(matrix.get(i, j) + " vs " + referenceMatrix.get(i, j));
                        check = false;
                        break;
                    }
                }
                if (!check) {
                    return false;
                }
            }
            return check;
        }

        private static void validate(int run, FP16Matrix matrix, FP16Matrix referenceMatrix) {
            if (run == 0) {
                System.out.println(" -- Result Correct? " + Multiplication.verify(matrix, referenceMatrix));
            } else {
                System.out.println();
            }
        }

        private static boolean verify(HalfFloatArray matrix, FP16Matrix referenceMatrix, int m, int n) {
            boolean check = true;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (Math.abs(matrix.get(i * m + j).getFloat32() - referenceMatrix.get(i, j)) > 0.1f) {
                        System.out.println(matrix.get(i * m + j) + " vs " + referenceMatrix.get(i, j));
                        check = false;
                        break;
                    }
                }
                if (!check) {
                    return false;
                }
            }
            return check;
        }

        private static void validate(int run, HalfFloatArray matrix, FP16Matrix referenceMatrix, int size) {
            if (run == 0) {
                System.out.println(" -- Result Correct? " + Multiplication.verify(matrix, referenceMatrix, size, size));
            } else {
                System.out.println();
            }
        }
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {

        FP16Matrix matrixA;
        FP16Matrix matrixB;

        // Matrix for results
        FP16Matrix matrixC;
        FP16Matrix matrixD;
        FP16Matrix matrixE;
        FP16Matrix matrixF;
        FP16Matrix matrixG;

        HalfFloatArray tma;
        HalfFloatArray tmb;
        HalfFloatArray resultTornadoVM;
        TaskGraph taskGraph;
        TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            // Using Panama Segments
            final int size = 1024;
            matrixA = new FP16Matrix(size, size);
            matrixB = new FP16Matrix(size, size);

            // Matrix for results
            matrixC = new FP16Matrix(size, size);
            matrixD = new FP16Matrix(size, size);
            matrixE = new FP16Matrix(size, size);
            matrixF = new FP16Matrix(size, size);
            matrixG = new FP16Matrix(size, size);

            matrixA.initRandom();
            matrixB.initRandom();

            // TornadoVM
            tma = Multiplication.transformMatrixForTornadoVM(matrixA);
            tmb = Multiplication.transformMatrixForTornadoVM(matrixB);
            resultTornadoVM = new HalfFloatArray(size * size);
            taskGraph = Multiplication.createTaskGraph(tma, tmb, resultTornadoVM, size);
            executionPlan = new TornadoExecutionPlan(taskGraph.snapshot());
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmSequential(JMHBenchmark state) {
            MatrixMultiplicationFP16.Multiplication.mxmSequential(state.matrixA, state.matrixB, state.matrixC);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmParallelStreams(JMHBenchmark state) {
            MatrixMultiplicationFP16.Multiplication.mxmParallelStreams(state.matrixA, state.matrixB, state.matrixD);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmParallelThreads(JMHBenchmark state) throws InterruptedException {
            MatrixMultiplicationFP16.Multiplication.mxmParallelThreads(state.matrixA, state.matrixB, state.matrixE);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmSequentialVectorized(JMHBenchmark state) {
            MatrixMultiplicationFP16.Multiplication.mxmSequentialVectorized(state.matrixA, state.matrixB, state.matrixF);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmParallelVectorized(JMHBenchmark state) {
            MatrixMultiplicationFP16.Multiplication.mxmParallelVectorized(state.matrixA, state.matrixB, state.matrixG);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmTornadoVM(JMHBenchmark state) {
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
                .include(MatrixMultiplicationFP16.class.getName() + ".*") //
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

        // Using Panama Segments
        FP16Matrix matrixA = new FP16Matrix(size, size);
        FP16Matrix matrixB = new FP16Matrix(size, size);

        // Matrix for results
        FP16Matrix outputReference = new FP16Matrix(size, size);
        FP16Matrix matrixD = new FP16Matrix(size, size);
        FP16Matrix matrixE = new FP16Matrix(size, size);
        FP16Matrix matrixF = new FP16Matrix(size, size);
        FP16Matrix matrixG = new FP16Matrix(size, size);

        matrixA.initRandom();
        matrixB.initRandom();

        // 6 implementations to compare
        ArrayList<ArrayList<Long>> timers = IntStream.range(0, 6) //
                .<ArrayList<Long>>mapToObj(i -> new ArrayList<>()) //
                .collect(Collectors.toCollection(ArrayList::new));

        // 1. Sequential
        for (int i = 0; i < Config.RUNS; i++) {
            long start = System.nanoTime();
            Multiplication.mxmSequential(matrixA, matrixB, outputReference);
            long end = System.nanoTime();
            long elapsedTime = (end - start);
            timers.get(0).add(elapsedTime);
            double elapsedTimeMilliseconds = elapsedTime * 1E-6;

            double gigaFlops = (1.0E-9 * FLOP) / (elapsedTime / TIME_SCALE_SECS);
            String formatGPUFGlops = String.format("%.2f", gigaFlops);

            System.out.println("Elapsed time: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- " + formatGPUFGlops + " GFLOP/s");

            if (option == Option.TORNADO_ONLY) {
                // We only run one iteration just to run the reference implementation to check results.
                break;
            }
        }

        if (option == Option.ALL || option == Option.JAVA_ONLY) {
            // 2. Parallel Streams
            for (int i = 0; i < Config.RUNS; i++) {
                long start = System.nanoTime();
                Multiplication.mxmParallelStreams(matrixA, matrixB, matrixD);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(1).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                double gigaFlops = (1.0E-9 * FLOP) / (elapsedTime / TIME_SCALE_SECS);
                String formatGPUFGlops = String.format("%.2f", gigaFlops);

                System.out.print("Stream Elapsed time: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- " + formatGPUFGlops + " GFLOP/s");
                Multiplication.validate(i, matrixD, outputReference);
            }

            // 3. Parallel with Java Threads
            for (int i = 0; i < Config.RUNS; i++) {
                long start = System.nanoTime();
                Multiplication.mxmParallelThreads(matrixA, matrixB, matrixE);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(2).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                double gigaFlops = (1.0E-9 * FLOP) / (elapsedTime / TIME_SCALE_SECS);
                String formatGPUFGlops = String.format("%.2f", gigaFlops);

                System.out.print("Elapsed time Threads: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- " + formatGPUFGlops + " GFLOP/s");
                Multiplication.validate(i, matrixE, outputReference);
            }

            // 4. Sequential Using the Vector API
            FP16Matrix bTranspose = Multiplication.transposeMatrix(matrixB);
            for (int i = 0; i < Config.RUNS; i++) {
                try {
                    long start = System.nanoTime();
                    Multiplication.mxmSequentialVectorized(matrixA, bTranspose, matrixF);
                    long end = System.nanoTime();
                    long elapsedTime = (end - start);
                    timers.get(3).add(elapsedTime);
                    double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                    double gigaFlops = (1.0E-9 * FLOP) / (elapsedTime / TIME_SCALE_SECS);
                    String formatGPUFGlops = String.format("%.2f", gigaFlops);

                    System.out.print("Elapsed time Vectorized: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- " + formatGPUFGlops + " GFLOP/s");
                    Multiplication.validate(i, matrixF, outputReference);
                } catch (UnsupportedOperationException e) {
                    timers.get(3).add(-1L);
                }
            }

            // 5. Parallel Streams using the Vector API
            for (int i = 0; i < Config.RUNS; i++) {
                try {
                    long start = System.nanoTime();
                    Multiplication.mxmParallelVectorized(matrixA, bTranspose, matrixG);
                    long end = System.nanoTime();
                    long elapsedTime = (end - start);
                    timers.get(4).add(elapsedTime);
                    double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                    double gigaFlops = (1.0E-9 * FLOP) / (elapsedTime / TIME_SCALE_SECS);
                    String formatGPUFGlops = String.format("%.2f", gigaFlops);

                    System.out.print("Elapsed time Parallel Vectorized: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- " + formatGPUFGlops + " GFLOP/s");
                    Multiplication.validate(i, matrixG, outputReference);
                } catch (UnsupportedOperationException e) {
                    timers.get(4).add(-1L);
                }
            }
        }

        if (option == Option.ALL || option == Option.TORNADO_ONLY) {
            // TornadoVM
            HalfFloatArray tma = Multiplication.transformMatrixForTornadoVM(matrixA);
            HalfFloatArray tmb = Multiplication.transformMatrixForTornadoVM(matrixB);
            HalfFloatArray resultTornadoVM = new HalfFloatArray(size * size);
            TaskGraph taskGraph = Multiplication.createTaskGraph(tma, tmb, resultTornadoVM, size);

            try(TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot())) {
                TornadoDevice device = TornadoExecutionPlan.getDevice(0, 0);
                WorkerGrid workerGrid = new WorkerGrid2D(size, size);
                workerGrid.setLocalWork(8, 8, 1);
                GridScheduler gridScheduler = new GridScheduler("benchmark.mxm", workerGrid);
                executionPlan
                        //.withGridScheduler(gridScheduler)
                        .withDevice(device);

                // 6. On the GPU using TornadoVM
                for (int i = 0; i < Config.RUNS; i++) {
                    long start = System.nanoTime();
                    executionPlan.execute();
                    long end = System.nanoTime();
                    long elapsedTime = (end - start);
                    timers.get(5).add(elapsedTime);
                    double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                    double gigaFlops = (1.0E-9 * FLOP) / (elapsedTime / TIME_SCALE_SECS);
                    String formatGPUFGlops = String.format("%.2f", gigaFlops);

                    System.out.print("Elapsed time TornadoVM-GPU: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- " + formatGPUFGlops + " GFLOP/s");
                    Multiplication.validate(i, resultTornadoVM, outputReference, size);
                }
            } catch (TornadoExecutionPlanException e) {
                throw new RuntimeException(e);
            }
        }
        if (option == Option.ALL) {
            Utils.dumpPerformanceTable(timers, 6, "mxm", Config.HEADER);
        }
    }

    @Override
    String getName() {
        return "matrix-multiplication-fp16";
    }

    @Override
    String printSize() {
        return getSize() + "x" + getSize();
    }

    public static void main(String[] args) throws InterruptedException {
        MatrixMultiplicationFP16 benchmark = new MatrixMultiplicationFP16(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixMul).size());
        benchmark.run(args);
    }
}
