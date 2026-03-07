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
import tornadovm.benchmarks.utils.Catalog;
import tornadovm.benchmarks.utils.Range;
import tornadovm.benchmarks.utils.Utils;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DFloat;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * How to run?
 * <code>
 *     tornado --jvm="-Dtornado.device.memory=2GB" -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.benchmarks.MatrixVector
 * </code>
 */
public class MatrixVector extends BenchmarkDriver {

    private int size;

    // ── inputs (immutable across iterations) ─────────────────────────────────
    private FloatMatrix matrix;
    private FVector vector;

    // ── outputs ──────────────────────────────────────────────────────────────
    private FVector outputReference;  // written by computeSequential()
    private FVector outputCpu;        // written by all parallel CPU backends

    // ── TornadoVM state ──────────────────────────────────────────────────────
    private Matrix2DFloat tma;
    private FloatArray tvector;
    private FloatArray resultTornadoVM;
    private boolean tornadoVMActive = false;

    public MatrixVector(int size) {
        this.size = size;
        matrix = new FloatMatrix(size, size);
        vector = new FVector(size);
        matrix.initRandom();
        vector.initRandom();
        outputReference = new FVector(size);
        outputCpu = new FVector(size);
        tma = Multiplication.transformMatrixForTornadoVM(matrix);
        tvector = Multiplication.transformFVectorForTornadoVM(vector);
        resultTornadoVM = new FloatArray(size);
    }

    // ── BenchmarkDriver abstract methods ─────────────────────────────────────

    @Override
    public void computeSequential() {
        Multiplication.mxvSequential(matrix, vector, outputReference);
    }

    @Override
    public void computeWithJavaStreams() {
        Multiplication.mxvParallelStreams(matrix, vector, outputCpu);
    }

    @Override
    public void computeWithJavaThreads() throws InterruptedException {
        Multiplication.mxvParallelThreads(matrix, vector, outputCpu);
    }

    @Override
    protected void computeWithJavaThreadsReusing(ExecutorService executor) throws InterruptedException {
        Range[] ranges = Utils.createRangesForCPU(matrix.M());
        List<Future<?>> futures = new ArrayList<>(ranges.length);
        for (int t = 0; t < ranges.length; t++) {
            final int idx = t;
            futures.add(executor.submit(() -> {
                for (int i = ranges[idx].min(); i < ranges[idx].max(); i++) {
                    float acc = 0;
                    for (int j = 0; j < vector.size(); j++) {
                        acc += matrix.get(i, j) * vector.get(j);
                    }
                    outputCpu.set(i, acc);
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
        Multiplication.mxvParallelVectorized(matrix, vector, outputCpu);
    }

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        tornadoVMActive = true;
        TaskGraph taskGraph = Multiplication.createTaskGraph(tma, tvector, resultTornadoVM);
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
                System.out.println(" -- Result Correct? " + Multiplication.verify(resultTornadoVM, outputReference));
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

    private static class FVector {

        private static final int FLOAT_SIZE = 4;
        private final MemorySegment segment;
        private int size;

        public FVector(int size) {
            this.size = size;
            final long segmentByteSize = size * FLOAT_SIZE;
            segment = Arena.ofAuto().allocate(segmentByteSize, 64);
        }

        public void set(int i, float value) {
            segment.set(JAVA_FLOAT, i * FLOAT_SIZE, value);
        }

        public float get(int i) {
            return segment.get(JAVA_FLOAT, i * FLOAT_SIZE);
        }

        public void initRandom() {
            Random r = new Random(71);
            for (int i = 0; i < size; i++) {
                set(i, r.nextFloat());
            }
        }

        public int size() {
            return size;
        }
    }

    private static class Multiplication {

        public static void mxvSequential(FloatMatrix a, FVector b, FVector c) {
            for (int i = 0; i < a.M(); i++) {
                float acc = 0;
                for (int j = 0; j < a.N(); j++) {
                    acc += a.get(i, j) * b.get(j);
                }
                c.set(i, acc);
            }
        }

        public static void mxvParallelStreams(FloatMatrix a, FVector b, FVector c) {
            IntStream.range(0, a.M()).parallel().forEach(i -> {
                float acc = 0;
                for (int j = 0; j < b.size; j++) {
                    acc += a.get(i, j) * b.get(j);
                }
                c.set(i, acc);
            });
        }

        public static void mxvParallelThreads(FloatMatrix a, FVector b, FVector c) throws InterruptedException {
            int maxProcessors = Runtime.getRuntime().availableProcessors();
            Range[] ranges = Utils.createRangesForCPU(a.M());
            Thread[] threads = new Thread[maxProcessors];
            IntStream.range(0, threads.length).forEach(t -> {
                threads[t] = new Thread(() -> {
                    for (int i = ranges[t].min(); i < ranges[t].max(); i++) {
                        float acc = 0;
                        for (int j = 0; j < b.size; j++) {
                            acc += a.get(i, j) * b.get(j);
                        }
                        c.set(i, acc);
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

        static final int FLOAT_BYTES = 4;
        public static void mxvSequentialVectorized(FloatMatrix a, FVector b, FVector c) {
            VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
            for (int i = 0; i < a.M(); i++) {
                float acc = 0;
                for (int j = 0; j < b.size; j += species.length()) {
                    FloatVector vector1 = FloatVector.fromMemorySegment(species, a.segment, (i * a.M() + j) * FLOAT_BYTES, ByteOrder.nativeOrder());
                    FloatVector vector2 = FloatVector.fromMemorySegment(species, b.segment, j * FLOAT_BYTES, ByteOrder.nativeOrder());
                    acc += vector1.mul(vector2).reduceLanes(VectorOperators.ADD);
                }
                c.set(i, acc);
            }
        }

        public static void mxvParallelVectorized(FloatMatrix a, FVector b, FVector c) {
            VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
            IntStream.range(0, a.M()).parallel().forEach(i -> {
                float acc = 0;
                for (int j = 0; j < b.size; j += species.length()) {
                    FloatVector vector1 = FloatVector.fromMemorySegment(species, a.segment, (i * a.M() + j) * FLOAT_BYTES, ByteOrder.nativeOrder());
                    FloatVector vector2 = FloatVector.fromMemorySegment(species, b.segment, j * FLOAT_BYTES, ByteOrder.nativeOrder());
                    acc += vector1.mul(vector2).reduceLanes(VectorOperators.ADD);
                }
                c.set(i, acc);
            });
        }

        private static void mxvTornadoVM(Matrix2DFloat a, FloatArray b, FloatArray c, final int size) {
            for (@Parallel int i = 0; i < a.getNumRows(); i++) {
                float sum = 0.0f;
                for (int j = 0; j < b.getSize(); j++) {
                    sum += a.get(i, j) * b.get(j);
                }
                c.set(i, sum);
            }
        }

        public static Matrix2DFloat transformMatrixForTornadoVM(FloatMatrix a) {
            int m = a.M();
            int n = a.N();
            Matrix2DFloat matrix2DFloat = new Matrix2DFloat(m, n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    matrix2DFloat.set(i, j, a.get(i, j));
                }
            }
            return matrix2DFloat;
        }

        public static FloatArray transformFVectorForTornadoVM(FVector a) {
            int m = a.size;
            FloatArray array = new FloatArray(m);
            IntStream.range(0, m).forEach(i -> array.set(i, a.get(i)));
            return array;
        }

        private static TaskGraph createTaskGraph(Matrix2DFloat a, FloatArray b, FloatArray c) {
            TaskGraph taskGraph = new TaskGraph("benchmark");
            taskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b) //
                    .task("mxv", Multiplication::mxvTornadoVM, a, b, c, a.getNumRows()) //
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, c);
            return taskGraph;
        }

        private static boolean verify(FVector array, FVector refArray) {
            boolean check = true;
            for (int i = 0; i < array.size(); i++) {
                if (Math.abs(array.get(i) - refArray.get(i)) > 0.1f) {
                    System.out.println(array.get(i) + " vs " + refArray.get(i));
                    check = false;
                    break;
                }
            }
            return check;
        }

        private static boolean verify(FloatArray array, FVector refArray) {
            boolean check = true;
            for (int i = 0; i < array.getSize(); i++) {
                if (Math.abs(array.get(i) - refArray.get(i)) > 0.1f) {
                    System.out.println(array.get(i) + " vs " + refArray.get(i));
                    check = false;
                    break;
                }
            }
            return check;
        }
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {

        FloatMatrix matrixA;
        FVector vector;
        FVector output;

        Matrix2DFloat tma;
        FloatArray tvector;
        FloatArray resultTornadoVM;
        TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            // Using Panama Segments
            final int size = 1024;
            matrixA = new FloatMatrix(size, size);
            vector = new FVector(size);
            output = new FVector(size);

            matrixA.initRandom();
            vector.initRandom();

            // TornadoVM
            tma = Multiplication.transformMatrixForTornadoVM(matrixA);
            tvector = Multiplication.transformFVectorForTornadoVM(vector);
            resultTornadoVM = new FloatArray(size);
            executionPlan = new TornadoExecutionPlan(Multiplication.createTaskGraph(tma, tvector, resultTornadoVM).snapshot());
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmSequential(JMHBenchmark state) {
            MatrixVector.Multiplication.mxvSequential(state.matrixA, state.vector, state.output);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmParallelStreams(JMHBenchmark state) {
            MatrixVector.Multiplication.mxvParallelStreams(state.matrixA, state.vector, state.output);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmParallelThreads(JMHBenchmark state) throws InterruptedException {
            MatrixVector.Multiplication.mxvParallelThreads(state.matrixA, state.vector, state.output);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmSequentialVectorized(JMHBenchmark state) {
            MatrixVector.Multiplication.mxvSequentialVectorized(state.matrixA, state.vector, state.output);
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void mxmParallelVectorized(JMHBenchmark state) {
            MatrixVector.Multiplication.mxvParallelVectorized(state.matrixA, state.vector, state.output);
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
                .include(MatrixVector.class.getName() + ".*") //
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
        return "matrix-vector";
    }

    @Override
    public String printSize() {
        return getSize() + "x" + getSize();
    }

    public static void main(String[] args) throws InterruptedException {
        MatrixVector benchmark = new MatrixVector(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixVector).size());
        benchmark.run(args);
    }
}
