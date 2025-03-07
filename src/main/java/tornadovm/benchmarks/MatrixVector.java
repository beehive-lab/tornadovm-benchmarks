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
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
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
 *     tornado --jvm="-Dtornado.device.memory=2GB" -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.MatrixVector
 * </code>
 */
public class MatrixVector extends Benchmark {

    private int size;
    public MatrixVector(int size) {
        this.size = size;
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
            if (!check) {
                return false;
            }
            return check;
        }

        private static void validate(int run, FVector array, FVector refArray) {
            if (run == 0) {
                System.out.println(" -- Result Correct? " + Multiplication.verify(array, refArray));
            } else {
                System.out.println();
            }
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
            if (!check) {
                return false;
            }
            return check;
        }

        private static void validate(int run, FloatArray array, FVector refArray) {
            if (run == 0) {
                System.out.println(" -- Result Correct? " + Multiplication.verify(array, refArray));
            } else {
                System.out.println();
            }
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
    int getSize() {
        return size;
    }

    @Override
    void runWithJMH() throws RunnerException {
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
    void runTestAll(final int size, Option option) throws InterruptedException {

        // Using Panama Segments
        FloatMatrix matrix = new FloatMatrix(size, size);
        FVector vector = new FVector(size);

        // Matrix for results
        FVector outputReference = new FVector(size);
        FVector outD = new FVector(size);
        FVector outE = new FVector(size);
        FVector outF = new FVector(size);
        FVector outG = new FVector(size);

        matrix.initRandom();
        vector.initRandom();

        // 6 implementations to compare
        ArrayList<ArrayList<Long>> timers = IntStream.range(0, 6) //
                .<ArrayList<Long>>mapToObj(i -> new ArrayList<>()) //
                .collect(Collectors.toCollection(ArrayList::new));

        // 1. Sequential
        for (int i = 0; i < Config.RUNS; i++) {
            long start = System.nanoTime();
            Multiplication.mxvSequential(matrix, vector, outputReference);
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
                Multiplication.mxvParallelStreams(matrix, vector, outD);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(1).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Stream Elapsed time: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                Multiplication.validate(i, outD, outputReference);
            }

            // 3. Parallel with Java Threads
            for (int i = 0; i < Config.RUNS; i++) {
                long start = System.nanoTime();
                Multiplication.mxvParallelThreads(matrix, vector, outE);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(2).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Elapsed time Threads: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                Multiplication.validate(i, outE, outputReference);
            }

            // 4. Sequential Using the Vector API
            for (int i = 0; i < Config.RUNS; i++) {
                long start = System.nanoTime();
                Multiplication.mxvSequentialVectorized(matrix, vector, outF);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(3).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Elapsed time Vectorized: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                Multiplication.validate(i, outF, outputReference);
            }

            // 5. Parallel Streams using the Vector API
            for (int i = 0; i < Config.RUNS; i++) {
                long start = System.nanoTime();
                Multiplication.mxvParallelVectorized(matrix, vector, outG);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(4).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Elapsed time Parallel Vectorized: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                Multiplication.validate(i, outG, outputReference);
            }
        }

        if (option == Option.ALL || option == Option.TORNADO_ONLY) {
            // TornadoVM
            Matrix2DFloat tma = Multiplication.transformMatrixForTornadoVM(matrix);
            FloatArray tmb = Multiplication.transformFVectorForTornadoVM(vector);
            FloatArray resultTornadoVM = new FloatArray(size);
            TaskGraph taskGraph = Multiplication.createTaskGraph(tma, tmb, resultTornadoVM);
            try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot())) {

                // 6. On the GPU using TornadoVM
                for (int i = 0; i < Config.RUNS; i++) {
                    long start = System.nanoTime();
                    executionPlan.execute();
                    long end = System.nanoTime();
                    long elapsedTime = (end - start);
                    timers.get(5).add(elapsedTime);
                    double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                    System.out.print("Elapsed time TornadoVM-GPU: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                    Multiplication.validate(i, resultTornadoVM, outputReference);
                }
            } catch (TornadoExecutionPlanException e) {
                throw new RuntimeException(e);
            }
        }
        if (option == Option.ALL) {
            Utils.dumpPerformanceTable(timers, 6, "matrixVector", Config.HEADER);
        }
    }

    @Override
    String getName() {
        return "matrix-vector";
    }

    @Override
    String printSize() {
        return getSize() + "x" + getSize();
    }

    public static void main(String[] args) throws InterruptedException {
        MatrixVector benchmark = new MatrixVector(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixVector).size());
        benchmark.run(args);
    }
}
