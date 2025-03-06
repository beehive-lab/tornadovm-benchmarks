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
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.ByteOrder;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class Saxpy extends BenchmarkDriver {

    private final int size;
    private float alpha = 0.2f;
    FloatArray arrayA;
    FloatArray arrayB;
    FloatArray output;
    FloatArray outputRef;

    public Saxpy(int size) {
        this.size = size;
        output = new FloatArray(size);
        outputRef = new FloatArray(size);
        arrayA = new FloatArray(size);
        arrayB = new FloatArray(size);
        Random r = new Random();
        for (int i = 0; i < size; i++) {
            arrayA.set(i, r.nextFloat());
            arrayB.set(i, r.nextFloat());
        }
    }

    @Override
    public void computeSequential() {
        for (int i = 0; i < size; i++) {
            outputRef.set(i, alpha * arrayA.get(i) + arrayB.get(i));
        }
    }

    @Override
    public void computeWithJavaStreams() {
        IntStream.range(0, size).forEach(i -> {
            output.set(i, alpha * arrayA.get(i) + arrayB.get(i));
        });
    }

    @Override
    public void computeWithJavaThreads() throws InterruptedException {
        Range[] ranges = Utils.createRangesForCPU(output.getSize());
        final int maxProcessors = Runtime.getRuntime().availableProcessors();

        Thread[] threads = new Thread[maxProcessors];
        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                for (int j = ranges[t].min(); j < ranges[t].max(); j++) {
                    output.set(j, alpha * arrayA.get(j) + arrayB.get(j));
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

    @Override
    public void computeWithParallelVectorAPI() {
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        final long FLOAT_BYES = 4;
        for (int i = 0; i < size; i += species.length()) {
            FloatVector a = FloatVector.fromMemorySegment(species, arrayA.getSegment(), i * FLOAT_BYES, ByteOrder.nativeOrder());
            FloatVector b = FloatVector.fromMemorySegment(species, arrayB.getSegment(), i * FLOAT_BYES, ByteOrder.nativeOrder());
            FloatVector add = a.mul(alpha).add(b);
            add.intoMemorySegment(output.getSegment(), i * FLOAT_BYES, ByteOrder.nativeOrder());
        }
    }

    public void computeWithTornadoVM(float alpha, FloatArray arrayA, FloatArray arrayB, FloatArray output) {
        for (@Parallel int i = 0; i < arrayA.getSize(); i++) {
            output.set(i, alpha * arrayA.get(i) + arrayB.get(i));
        }
    }

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        TaskGraph taskGraph = new TaskGraph("benchmark")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, arrayA, arrayB)
                .task("saxpy", this::computeWithTornadoVM, alpha, arrayA, arrayB, output)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);
        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    @Override
    public void resetOutputs() {
        output.clear();
    }

    private boolean validate(FloatArray outputRef, FloatArray output) {
        for (int i = 0; i < outputRef.getSize(); i++) {
            if (Math.abs(outputRef.get(i) - output.get(i)) > Config.DELTA) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void validate(int run) {
        if (run == 0) {
            System.out.println(" -- Result Correct? " + validate(outputRef, output));
        } else {
            System.out.println();
        }
    }

    @Override
    int getSize() {
        return size;
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {

        private Saxpy saxpy;
        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            saxpy = new Saxpy(Catalog.DEFAULT.get(Catalog.BenchmarkID.Saxpy).size());
            executionPlan = saxpy.buildExecutionPlan();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void saxpySequential(JMHBenchmark state) {
            state.saxpy.computeSequential();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void saxpyParallelStreams(JMHBenchmark state) {
            state.saxpy.computeWithJavaStreams();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void saxpyParallelThreads(JMHBenchmark state) {
            try {
                state.saxpy.computeWithJavaThreads();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void saxpyParallelVectorAPI(JMHBenchmark state) {
            state.saxpy.computeWithParallelVectorAPI();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void saxpyTornadoVM(JMHBenchmark state) {
            state.executionPlan.execute();
        }
    }

    @Override
    void runWithJMH() throws RunnerException {
        org.openjdk.jmh.runner.options.Options opt = new OptionsBuilder() //
                .include(Saxpy.class.getName() + ".*") //
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
        return "saxpy";
    }

    @Override
    String printSize() {
        return getSize() + "";
    }

    public static void main(String[] args) throws InterruptedException {
        Saxpy benchmark = new Saxpy(Catalog.DEFAULT.get(Catalog.BenchmarkID.Montecarlo).size());
        benchmark.run(args);
    }
}
