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
import tornadovm.benchmarks.utils.Catalog;
import tornadovm.benchmarks.utils.Config;
import tornadovm.benchmarks.utils.Range;
import tornadovm.benchmarks.utils.Utils;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class Silu extends BenchmarkDriver {

    private final int size;
    FloatArray shb;
    FloatArray shbRef;
    FloatArray shb2;
    FloatArray shbInit;

    public Silu(int size) {
        this.size = size;
        shb = new FloatArray(size);
        shb2 = new FloatArray(size);
        shbRef = new FloatArray(size);
        shbInit = new FloatArray(size);
        Random r = new Random();
        for (int i = 0; i < size; i++) {
            shb2.set(i, r.nextFloat());
            shbInit.set(i, r.nextFloat());
            shbRef.set(i, shbInit.get(i));
        }
        init();
    }

    private void init() {
        IntStream.range(0, size).forEach(i -> shb.set(i, shbInit.get(i)));
    }

    @Override
    public void computeSequential() {
        for (int i = 0; i < size; i++) {
            float val = shb.get(i);
            val *= (1.0f / (1.0f + TornadoMath.exp(-val)));
            val *= shb2.get(i);
            shbRef.set(i, val);
        }
    }

    @Override
    public void computeWithJavaStreams() {
        IntStream.range(0, size).parallel().forEach(i -> {
            float val = shb.get(i);
            val *= (1.0f / (1.0f + TornadoMath.exp(-val)));
            val *= shb2.get(i);
            shb.set(i, val);
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
                    shb.set(j, val);
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

    /**
     * Reuses the shared thread pool supplied by {@link BenchmarkDriver} to avoid
     * per-iteration thread-creation overhead in the timed region.
     */
    @Override
    protected void computeWithJavaThreadsReusing(ExecutorService executor) throws InterruptedException {
        Range[] ranges = Utils.createRangesForCPU(shb.getSize());
        List<Future<?>> futures = new ArrayList<>(ranges.length);
        for (int t = 0; t < ranges.length; t++) {
            final int idx = t;
            futures.add(executor.submit(() -> {
                for (int j = ranges[idx].min(); j < ranges[idx].max(); j++) {
                    float val = shb.get(j);
                    val *= (1.0f / (1.0f + TornadoMath.exp(-val)));
                    val *= shb2.get(j);
                    shb.set(j, val);
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
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        final long FLOAT_BYES = 4;
        final int loopBound = species.loopBound(size);
        int i = 0;
        for (; i < loopBound; i += species.length()) {
            FloatVector vA = FloatVector.fromMemorySegment(species, shb.getSegment(), i * FLOAT_BYES, ByteOrder.nativeOrder());
            FloatVector vB = FloatVector.fromMemorySegment(species, shb2.getSegment(), i * FLOAT_BYES, ByteOrder.nativeOrder());
            // 4th-order Taylor approximation for exp(-vA): 1 - vA + vA²/2! - vA³/3! + vA⁴/4!
            FloatVector vA2 = vA.mul(vA);
            FloatVector vA3 = vA2.mul(vA);
            FloatVector vA4 = vA3.mul(vA);
            Vector<Float> one = FloatVector.broadcast(species, 1.0f);
            Vector<Float> resultExp = one
                    .sub(vA)
                    .add(vA2.mul(0.5f))
                    .sub(vA3.mul(1.0f / 6.0f))
                    .add(vA4.mul(1.0f / 24.0f));

            // silu: vA * sigmoid(vA) * vB  where sigmoid(x) = 1 / (1 + exp(-x))
            Vector<Float> divB = one.add(resultExp);
            Vector<Float> valDiv = one.div(divB);
            valDiv = valDiv.mul(vA).mul(vB);
            valDiv.intoMemorySegment(shb.getSegment(), i * FLOAT_BYES, ByteOrder.nativeOrder());
        }
        for (; i < size; i++) {
            float val = shb.get(i);
            val *= (1.0f / (1.0f + TornadoMath.exp(-val)));
            val *= shb2.get(i);
            shb.set(i, val);
        }
    }

    private static void computeWithTornadoVM(int size, FloatArray shb, FloatArray shb2) {
        for (@Parallel int i = 0; i < size; i++) {
            float val = shb.get(i);
            val *= (1.0f / (1.0f + TornadoMath.exp(-val)));
            val *= shb2.get(i);
            shb.set(i, val);
        }
    }

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        init();
        TaskGraph taskGraph = new TaskGraph("benchmark")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, shb2)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, shb)
                .task("silu", Silu::computeWithTornadoVM, size, shb, shb2)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, shb);
        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    @Override
    public void resetOutputs() {
        init();
    }

    /** Package-private hook for unit tests: true iff the last parallel result matches the sequential reference. */
    boolean isResultCorrect() {
        return validate(shbRef, shb);
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
            System.out.println(" -- Result Correct? " + validate(shbRef, shb));
        } else {
            System.out.println();
        }
    }

    @Override
    public int getSize() {
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
    public void runWithJMH() throws RunnerException {
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
    public String getName() {
        return "silu";
    }

    @Override
    public String printSize() {
        return getSize() + "";
    }

    public static void main(String[] args) throws InterruptedException {
        Silu benchmark = new Silu(Catalog.DEFAULT.get(Catalog.BenchmarkID.Silu).size());
        benchmark.run(args);
    }
}
