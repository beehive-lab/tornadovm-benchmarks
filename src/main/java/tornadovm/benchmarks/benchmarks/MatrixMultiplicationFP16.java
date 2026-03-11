/*
 * Copyright (c) 2025-2026, APT Group, Department of Computer Science,
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
package tornadovm.benchmarks.benchmarks;

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
import tornadovm.benchmarks.utils.Catalog;
import tornadovm.benchmarks.utils.Range;
import tornadovm.benchmarks.utils.Utils;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * How to run?
 * <code>
 *     tornado -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.benchmarks.MatrixMultiplicationFP16
 * </code>
 */
public class MatrixMultiplicationFP16 extends BenchmarkDriver {

    int size;

    // ── inputs (immutable across iterations) ─────────────────────────────────
    private FP16Matrix matrixA;
    private FP16Matrix matrixB;

    // ── outputs ──────────────────────────────────────────────────────────────
    private FP16Matrix outputReference;  // written by computeSequential()
    private FP16Matrix outputCpu;        // written by all parallel CPU backends

    // ── TornadoVM state ──────────────────────────────────────────────────────
    private HalfFloatArray tma;
    private HalfFloatArray tmb;
    private HalfFloatArray resultTornadoVM;
    private boolean tornadoVMActive = false;

    public MatrixMultiplicationFP16(int size) {
        this.size = size;
        matrixA = new FP16Matrix(size, size);
        matrixB = new FP16Matrix(size, size);
        matrixA.initRandom();
        matrixB.initRandom();
        outputReference = new FP16Matrix(size, size);
        outputCpu = new FP16Matrix(size, size);
        tma = Multiplication.transformMatrixForTornadoVM(matrixA);
        tmb = Multiplication.transformMatrixForTornadoVM(matrixB);
        resultTornadoVM = new HalfFloatArray(size * size);
    }

    // ── BenchmarkDriver abstract methods ─────────────────────────────────────

    @Override
    public void computeSequential() {
        Multiplication.mxmSequential(matrixA, matrixB, outputReference);
    }

    @Override
    public void computeWithJavaStreams() {
        Multiplication.mxmParallelStreams(matrixA, matrixB, outputCpu);
    }

    @Override
    public void computeWithJavaThreads() throws InterruptedException {
        Multiplication.mxmParallelThreads(matrixA, matrixB, outputCpu);
    }

    @Override
    protected void computeWithJavaThreadsReusing(ExecutorService executor) throws InterruptedException {
        Range[] ranges = Utils.createRangesForCPU(size);
        List<Future<?>> futures = new ArrayList<>(ranges.length);
        for (int t = 0; t < ranges.length; t++) {
            final int idx = t;
            futures.add(executor.submit(() -> {
                for (int i = ranges[idx].min(); i < ranges[idx].max(); i++) {
                    for (int j = 0; j < size; j++) {
                        float acc = 0;
                        for (int k = 0; k < size; k++) {
                            acc += matrixA.get(i, k) * matrixB.get(k, j);
                        }
                        outputCpu.set(i, j, acc);
                    }
                }
            }));
        }
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (ExecutionException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public void computeWithParallelVectorAPI() {
        // FP16 Vector API is not supported; the driver catches the thrown exception and marks as failed.
        Multiplication.mxmParallelVectorized(matrixA, matrixB, outputCpu);
    }

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        tornadoVMActive = true;
        TaskGraph taskGraph = Multiplication.createTaskGraph(tma, tmb, resultTornadoVM, size);
        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    @Override
    public void resetOutputs() {
        // Inputs are immutable; every backend completely overwrites its output — no reset needed.
    }

    @Override
    public void validate(int run) {
        if (run == 0) {
            if (tornadoVMActive) {
                System.out.println(" -- Result Correct? " + Multiplication.verify(resultTornadoVM, outputReference, size, size));
            } else {
                System.out.println(" -- Result Correct? " + Multiplication.verify(outputCpu, outputReference));
            }
        } else {
            System.out.println();
        }
    }

    // ── Inner classes (computation kernels) ──────────────────────────────────

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
    public int getSize() {
        return size;
    }

    @Override
    public void runWithJMH() throws RunnerException {
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
    public String getName() {
        return "matrix-multiplication-fp16";
    }

    @Override
    public String printSize() {
        return getSize() + "x" + getSize();
    }

    public static void main(String[] args) throws InterruptedException {
        MatrixMultiplicationFP16 benchmark = new MatrixMultiplicationFP16(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixMul).size());
        benchmark.run(args);
    }
}
