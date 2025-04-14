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
import jdk.incubator.vector.Vector;
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
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.ByteOrder;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class Silu extends BenchmarkDriver {

    private final int size;
    FloatArray shb;
    FloatArray shb2;
    FloatArray shb2Ref;
    FloatArray shb2Init;

    public Silu(int size) {
        this.size = size;
        shb = new FloatArray(size);
        shb2 = new FloatArray(size);
        shb2Ref = new FloatArray(size);
        shb2Init = new FloatArray(size);
        Random r = new Random();
        for (int i = 0; i < size; i++) {
            shb.set(i, r.nextFloat());
            shb2Init.set(i, r.nextFloat());
            shb2Ref.set(i, shb2Init.get(i));
        }
        init();
    }

    private void init() {
        IntStream.range(0, size).forEach(i -> shb2.set(i, shb2Init.get(i)));
    }

    @Override
    public void computeSequential() {
        for (int i = 0; i < size; i++) {
            float val = shb.get(i);
            val *= (1.0f / (1.0f + TornadoMath.exp(-val)));
            val *= shb2Ref.get(i);
            shb2Ref.set(i, val);
        }
    }

    @Override
    public void computeWithJavaStreams() {
        IntStream.range(0, size).parallel().forEach(i -> {
            float val = shb.get(i);
            val *= (1.0f / (1.0f + TornadoMath.exp(-val)));
            val *= shb2.get(i);
            shb2.set(i, val);
        });
    }

    @Override
    public void computeWithJavaThreads() throws InterruptedException {
        Range[] ranges = Utils.createRangesForCPU(shb.getSize());
        final int maxProcessors = Runtime.getRuntime().availableProcessors();

        Thread[] threads = new Thread[maxProcessors];
        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                for (int j = ranges[t].min(); j < ranges[t].max(); j++) {
                    float val = shb.get(j);
                    val *= (1.0f / (1.0f + TornadoMath.exp(-val)));
                    val *= shb2.get(j);
                    shb2.set(j, val);
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
        final int loopBound = species.loopBound(size);
        int i = 0;
        for (; i < loopBound; i += species.length()) {
            FloatVector vA = FloatVector.fromMemorySegment(species, shb.getSegment(), i * FLOAT_BYES, ByteOrder.nativeOrder());
            FloatVector vB = FloatVector.fromMemorySegment(species, shb2.getSegment(), i * FLOAT_BYES, ByteOrder.nativeOrder());
            FloatVector mVA = vA.mul(-1.0f);

            // Compute exp(x) using the Taylor Approximation: exp(x) ~= 1 + x + x^2/2!
            Vector<Float> one = FloatVector.broadcast(species, 1.0f);
            Vector<Float> mul = vA.mul(vA);
            Vector<Float> half = FloatVector.broadcast(species, 0.5f);
            Vector<Float> resultExp =  one.add(mVA).add(mul.mul(half));

            Vector<Float> divB = one.add(resultExp);
            Vector<Float> valDiv = one.div(divB);
            valDiv = valDiv.mul(vB);
            valDiv.intoMemorySegment(shb2.getSegment(), i * FLOAT_BYES, ByteOrder.nativeOrder());
        }
        for (; i < size; i++) {
            float val = shb.get(i);
            val *= (1.0f / (1.0f + TornadoMath.exp(-val)));
            val *= shb2.get(i);
            shb2.set(i, val);
        }
    }

    private static void computeWithTornadoVM(int size, FloatArray shb, FloatArray shb2) {
        for (@Parallel int i = 0; i < size; i++) {
            float val = shb.get(i);
            val *= (1.0f / (1.0f + TornadoMath.exp(-val)));
            val *= shb2.get(i);
            shb2.set(i, val);
        }
    }

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        init();
        TaskGraph taskGraph = new TaskGraph("benchmark")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, shb)
                .task("silu", Silu::computeWithTornadoVM, size, shb, shb2)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, shb2);
        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    @Override
    public void resetOutputs() {
        init();
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
            System.out.println(" -- Result Correct? " + validate(shb2Ref, shb2));
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

        private Silu siluKernel;
        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            siluKernel = new Silu(Catalog.DEFAULT.get(Catalog.BenchmarkID.Silu).size());
            executionPlan = siluKernel.buildExecutionPlan();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void siluSequential(JMHBenchmark state) {
            state.siluKernel.computeSequential();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void siluParallelStreams(JMHBenchmark state) {
            state.siluKernel.computeWithJavaStreams();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void siluParallelThreads(JMHBenchmark state) {
            try {
                state.siluKernel.computeWithJavaThreads();
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
        public void siluParallelVectorAPI(JMHBenchmark state) {
            state.siluKernel.computeWithParallelVectorAPI();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void siluTornadoVM(JMHBenchmark state) {
            state.executionPlan.execute();
        }
    }

    @Override
    void runWithJMH() throws RunnerException {
        org.openjdk.jmh.runner.options.Options opt = new OptionsBuilder() //
                .include(Silu.class.getName() + ".*") //
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
        return "silu";
    }

    @Override
    String printSize() {
        return getSize() + "";
    }

    public static void main(String[] args) throws InterruptedException {
        Silu benchmark = new Silu(Catalog.DEFAULT.get(Catalog.BenchmarkID.Silu).size());
        benchmark.run(args);
    }
}
