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
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DFloat;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * How to run?
 * <code>
 *     tornado -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.MatrixMultiplication
 * </code>
 */
public class MatrixMultiplication extends Benchmark {

    int size;
    private double FLOP;
    private final static float TIME_SCALE_SECS = 1.0E09f;

    public MatrixMultiplication(int size) {
        this.size = size;
        FLOP = 2 * Math.pow(size, 3);
    }

    /**
     * Float MxN Matrix
     */
    private static class FloatMatrix {

        private static final int FLOAT_SIZE = 4;

        private final int m;
        private final int n;
        private final MemorySegment segment;

        public FloatMatrix(int m, int n) {
            this.m = m;
            this.n = n;
            final long segmentByteSize = n * m * FLOAT_SIZE;
            segment = Arena.ofAuto().allocate(segmentByteSize, 64);
        }

        public void set(int i, int j, float value) {
            final int index = i * m + j;
            segment.set(JAVA_FLOAT, index * FLOAT_SIZE, value);
        }

        public float get(int i, int j) {
            final int index = i * m + j;
            return segment.get(JAVA_FLOAT, index * FLOAT_SIZE);
        }

        public void initRandom() {
            Random r = new Random(71);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    set(i, j, r.nextFloat());
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
        public static void mxmSequential(FloatMatrix a, FloatMatrix b, FloatMatrix c) {
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

        public static void mxmParallelStreams(FloatMatrix a, FloatMatrix b, FloatMatrix c) {
            IntStream.range(0, a.M()).parallel().forEach(i -> IntStream.range(0, b.N()).parallel().forEach(j -> {
                float acc = 0;
                for (int k = 0; k < c.M(); k++) {
                    acc += a.get(i, k) * b.get(k, j);
                }
                c.set(i, j, acc);
            }));
        }

        public static void mxmParallelThreads(FloatMatrix a, FloatMatrix b, FloatMatrix c) throws InterruptedException {

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

        public static FloatMatrix transposeMatrix(FloatMatrix matrix) {
            FloatMatrix matrixTranspose = new FloatMatrix(matrix.M(), matrix.N());
            for (int i = 0; i < matrix.M(); i++) {
                for (int j = 0; j < matrix.N(); j++) {
                    matrixTranspose.set(i, j, matrix.get(j, i));
                }
            }
            return matrixTranspose;
        }

        static final int FLOAT_BYTES = 4;
        public static void mxmSequentialVectorized(FloatMatrix a, FloatMatrix b, FloatMatrix c) {
            VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
            for (int i = 0; i < a.M(); i++) {
                for (int j = 0; j < a.N(); j++) {
                    float acc = 0;
                    for (int k = 0; k < c.M(); k += species.length()) {
                        FloatVector vector1 = FloatVector.fromMemorySegment(species, a.segment, (i * a.M() + k) * FLOAT_BYTES, ByteOrder.nativeOrder());
                        FloatVector vector2 = FloatVector.fromMemorySegment(species, b.segment, (j * b.N() + k) * FLOAT_BYTES, ByteOrder.nativeOrder());
                        acc += vector1.mul(vector2).reduceLanes(VectorOperators.ADD);
                    }
                    c.set(i, j, acc);
                }
            }
        }

        public static void mxmParallelVectorized(FloatMatrix a, FloatMatrix b, FloatMatrix c) {
            VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
            IntStream.range(0, a.M()).parallel().forEach(i -> IntStream.range(0, b.N()).parallel().forEach(j -> {
                float acc = 0;
                for (int k = 0; k < c.M(); k += species.length()) {
                    FloatVector vector1 = FloatVector.fromMemorySegment(species, a.segment, (i * a.M() + k) * FLOAT_BYTES, ByteOrder.nativeOrder());
                    FloatVector vector2 = FloatVector.fromMemorySegment(species, b.segment, (j * b.N() + k) * FLOAT_BYTES, ByteOrder.nativeOrder());
                    acc += vector1.mul(vector2).reduceLanes(VectorOperators.ADD);
                }
                c.set(i, j, acc);
            }));
        }

        /**
         * This method computes squared matrix multiplication.
         * @param a
         * @param b
         * @param c
         * @param size (num rows and num columns)
         */
        private static void mxmTornadoVM(Matrix2DFloat a, Matrix2DFloat b, Matrix2DFloat c, final int size) {
            for (@Parallel int i = 0; i < size; i++) {
                for (@Parallel int j = 0; j < size; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < size; k++) {
                        sum += a.get(i, k) * b.get(k, j);
                    }
                    c.set(i, j, sum);
                }
            }
        }

        public static Matrix2DFloat transformMatrixForTornadoVM(FloatMatrix a) {
            final int m = a.M();
            final int n = a.N();
            final Matrix2DFloat matrix2DFloat = new Matrix2DFloat(m, n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    matrix2DFloat.set(i, j, a.get(i, j));
                }
            }
            return matrix2DFloat;
        }

        private static TornadoExecutionPlan createTornadoVMPlan(Matrix2DFloat a, Matrix2DFloat b, Matrix2DFloat c) {
            TaskGraph taskGraph = new TaskGraph("benchmark");
            taskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                    .task("mxm", Multiplication::mxmTornadoVM, a, b, c, a.getNumRows()) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, c);
            TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot());

            TornadoDevice device = TornadoExecutionPlan.getDevice(0, 0);

            WorkerGrid workerGrid = new WorkerGrid2D(a.getNumRows(), a.getNumColumns());
            workerGrid.setLocalWork(8, 8, 1);
            GridScheduler gridScheduler = new GridScheduler("benchmark.mxm", workerGrid);
            executionPlan
                    //.withGridScheduler(gridScheduler)
                    .withDevice(device);

            return executionPlan;
        }

        private static boolean verify(FloatMatrix matrix, FloatMatrix referenceMatrix) {
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

        private static void validate(int run, FloatMatrix matrix, FloatMatrix referenceMatrix) {
            if (run == 0) {
                System.out.println(" -- Result Correct? " + Multiplication.verify(matrix, referenceMatrix));
            } else {
                System.out.println();
            }
        }

        private static boolean verify(Matrix2DFloat matrix, FloatMatrix referenceMatrix) {
            boolean check = true;
            for (int i = 0; i < matrix.getNumRows(); i++) {
                for (int j = 0; j < matrix.getNumColumns(); j++) {
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

        private static void validate(int run, Matrix2DFloat matrix, FloatMatrix referenceMatrix) {
            if (run == 0) {
                System.out.println(" -- Result Correct? " + Multiplication.verify(matrix, referenceMatrix));
            } else {
                System.out.println();
            }
        }
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {

        FloatMatrix matrixA;
        FloatMatrix matrixB;

        // Matrix for results
        FloatMatrix matrixC;
        FloatMatrix matrixD;
        FloatMatrix matrixE;
        FloatMatrix matrixF;
        FloatMatrix matrixG;

        Matrix2DFloat tma;
        Matrix2DFloat tmb;
        Matrix2DFloat resultTornadoVM;
        TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            // Using Panama Segments
            final int size = 1024;
            matrixA = new FloatMatrix(size, size);
            matrixB = new FloatMatrix(size, size);

            // Matrix for results
            matrixC = new FloatMatrix(size, size);
            matrixD = new FloatMatrix(size, size);
            matrixE = new FloatMatrix(size, size);
            matrixF = new FloatMatrix(size, size);
            matrixG = new FloatMatrix(size, size);

            matrixA.initRandom();
            matrixB.initRandom();

            // TornadoVM
            tma = Multiplication.transformMatrixForTornadoVM(matrixA);
            tmb = Multiplication.transformMatrixForTornadoVM(matrixB);
            resultTornadoVM = new Matrix2DFloat(size, size);
            executionPlan = Multiplication.createTornadoVMPlan(tma, tmb, resultTornadoVM);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmSequential(JMHBenchmark state) {
            MatrixMultiplication.Multiplication.mxmSequential(state.matrixA, state.matrixB, state.matrixC);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmParallelStreams(JMHBenchmark state) {
            MatrixMultiplication.Multiplication.mxmParallelStreams(state.matrixA, state.matrixB, state.matrixD);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmParallelThreads(JMHBenchmark state) throws InterruptedException {
            MatrixMultiplication.Multiplication.mxmParallelThreads(state.matrixA, state.matrixB, state.matrixE);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmSequentialVectorized(JMHBenchmark state) {
            MatrixMultiplication.Multiplication.mxmSequentialVectorized(state.matrixA, state.matrixB, state.matrixF);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmParallelVectorized(JMHBenchmark state) {
            MatrixMultiplication.Multiplication.mxmParallelVectorized(state.matrixA, state.matrixB, state.matrixG);
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
                .include(MatrixMultiplication.class.getName() + ".*") //
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
        FloatMatrix matrixA = new FloatMatrix(size, size);
        FloatMatrix matrixB = new FloatMatrix(size, size);

        // Matrix for results
        FloatMatrix outputReference = new FloatMatrix(size, size);
        FloatMatrix matrixD = new FloatMatrix(size, size);
        FloatMatrix matrixE = new FloatMatrix(size, size);
        FloatMatrix matrixF = new FloatMatrix(size, size);
        FloatMatrix matrixG = new FloatMatrix(size, size);

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
            FloatMatrix bTranspose = Multiplication.transposeMatrix(matrixB);
            for (int i = 0; i < Config.RUNS; i++) {
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
            }

            // 5. Parallel Streams using the Vector API
            for (int i = 0; i < Config.RUNS; i++) {
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
            }
        }

        if (option == Option.ALL || option == Option.TORNADO_ONLY) {
            // TornadoVM
            Matrix2DFloat tma = Multiplication.transformMatrixForTornadoVM(matrixA);
            Matrix2DFloat tmb = Multiplication.transformMatrixForTornadoVM(matrixB);
            Matrix2DFloat resultTornadoVM = new Matrix2DFloat(size, size);
            TornadoExecutionPlan executionPlan = Multiplication.createTornadoVMPlan(tma, tmb, resultTornadoVM);

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
                Multiplication.validate(i, resultTornadoVM, outputReference);
            }
        }
        if (option == Option.ALL) {
            Utils.dumpPerformanceTable(timers, 6, "mxm", Config.HEADER);
        }
    }

    @Override
    String getName() {
        return "MatrixMultiplication";
    }

    @Override
    String printSize() {
        return getSize() + "x" + getSize();
    }

    public static void main(String[] args) throws InterruptedException {
        MatrixMultiplication benchmark = new MatrixMultiplication(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixMul).size());
        benchmark.run(args);
    }
}
