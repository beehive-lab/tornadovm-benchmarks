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
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.matrix.Matrix2DFloat;

import java.nio.ByteOrder;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * How to run?
 *
 * <code>
 *     tornado -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.MatrixTranspose
 * </code>
 */
public class MatrixTranspose extends BenchmarkDriver {

    final static int SIZE = 8192;
    Matrix2DFloat a;
    Matrix2DFloat refOutput;
    Matrix2DFloat output;

    public MatrixTranspose() {
        a = new Matrix2DFloat(SIZE, SIZE);
        refOutput = new Matrix2DFloat(SIZE, SIZE);
        output = new Matrix2DFloat(SIZE, SIZE);
        Random r = new Random(71);
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                a.set(i, j, r.nextFloat());
            }
        }
    }

    @Override
    public void computeSequential() {
        for (int i = 0; i < a.getNumRows(); i++) {
            for (int j = 0; j < refOutput.getNumColumns(); j++) {
                refOutput.set(j, i, a.get(i, j));
            }
        }
    }

    @Override
    public void computeWithJavaStreams() {
        IntStream.range(0, a.getNumRows()).parallel().forEach(i -> {
            IntStream.range(0, output.getNumColumns()).parallel().forEach(j -> {
                output.set(j, i, a.get(i, j));
            });
        });
    }

    @Override
    public void computeWithJavaThreads() throws InterruptedException {
        Range[] ranges = Utils.createRangesForCPU(SIZE);
        int maxProcessors = Runtime.getRuntime().availableProcessors();
        Thread[] threads = new Thread[maxProcessors];

        IntStream.range(0, threads.length).forEach(threadIndex -> {
            threads[threadIndex] = new Thread(() -> {
                for (int i = ranges[threadIndex].min(); i < ranges[threadIndex].max(); i++) {
                    for (int j = 0; j < output.getNumColumns(); j++) {
                        output.set(j, i, a.get(i, j));
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
        VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
        final int vectorLength = SPECIES.length();
        IntStream.range(0, a.getNumRows()).parallel().forEach(i -> {
            for (int j = 0; j < output.getNumColumns(); j += vectorLength) {
                FloatVector vector = FloatVector.fromMemorySegment(SPECIES, a.getSegment(), ((i * a.getNumRows()) + j) * 4, ByteOrder.nativeOrder());
                for (int k = 0; k < vectorLength; k++) {
                    if (j + k < output.getNumColumns()) {
                        output.set(j + k, i, vector.lane(k));
                    }
                }
            }
        });
    }

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        TaskGraph taskGraph = new TaskGraph("benchmark")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, a)
                .task("matrixTranspose", this::computeWithTornadoVM, a, output)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);
        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    @Override
    public void resetOutputs() {
        for (int i = 0; i < a.getNumRows(); i++) {
            for (int j = 0; j < a.getNumColumns(); j++) {
                output.set(i, j, 0);
            }
        }
    }

    @Override
    public void validate(int i ) {
        validate(i, output, refOutput);
    }

    public void computeWithTornadoVM(Matrix2DFloat a, Matrix2DFloat b) {
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
        if (run == 0) {
            System.out.println(" -- Result Correct? " + verify(matrix, reference));
        } else {
            System.out.println();
        }
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
            executionPlan = matrixTranspose.buildExecutionPlan();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void matrixTransposeSequential(JMHBenchmark state) {
            state.matrixTranspose.computeSequential();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void matrixTransposeParallelStreams(JMHBenchmark state) {
            state.matrixTranspose.computeWithJavaStreams();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void matrixTransposeParallelThreads(JMHBenchmark state) throws InterruptedException {
            state.matrixTranspose.computeWithJavaThreads();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void matrixTransposeParallelVectorAPI(JMHBenchmark state) {
            state.matrixTranspose.computeWithParallelVectorAPI();
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
    String getName() {
        return "MatrixTranspose";
    }

    @Override
    String printSize() {
        return getSize() + "x" + getSize();
    }
}
