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
import tornadovm.benchmarks.utils.Config;
import tornadovm.benchmarks.utils.Range;
import tornadovm.benchmarks.utils.Utils;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.annotations.Reduce;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * <p>How to run?
 * <code>
 *     tornado -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.benchmarks.SoftMax
 * </code>
 * </p>
 */
public class SoftMax extends BenchmarkDriver {

    private final int size;
    private FloatArray xRef;
    private FloatArray xInit;
    private FloatArray xSeq;  // isolated buffer for computeSequential(); never shared with parallel backends
    private FloatArray x;
    private FloatArray temp;
    private Float[] xStreams;
    boolean streams;

    public SoftMax(int size) {
        this.size = size;
        xRef = new FloatArray(size);
        xInit = new FloatArray(size);
        xSeq = new FloatArray(size);
        x = new FloatArray(size);
        temp = new FloatArray(1);
        xStreams = new Float[size];
        init();
        setInit();
        initReference();
    }

    private void initReference() {
        // Pre-compute xRef = softmax(xInit) exactly once at construction.
        // xRef is never modified again, so every backend validates against
        // a stable, correct reference regardless of execution order.
        float max_val = xRef.get(0);
        for (int i = 1; i < size; i++) {
            if (xRef.get(i) > max_val) max_val = xRef.get(i);
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            xRef.set(i, TornadoMath.exp(xRef.get(i) - max_val));
            sum += xRef.get(i);
        }
        for (int i = 0; i < size; i++) {
            xRef.set(i, xRef.get(i) / sum);
        }
    }

    private void init() {
        Random rand = new Random(71);
        IntStream.range(0, size).forEach(i -> {
            xInit.set(i, rand.nextFloat());
            xRef.set(i, xInit.get(i));
            xStreams[i] = xInit.get(i);
        });
    }

    private void setInit() {
        IntStream.range(0, size).forEach(i -> {
            xSeq.set(i, xInit.get(i));
            x.set(i, xInit.get(i));
            xStreams[i] = xInit.get(i);
        });
    }

    @Override
    public void computeSequential() {
        // find max value (for numerical stability)
        float max_val = xSeq.get(0);
        for (int i = 1; i < size; i++) {
            if (xSeq.get(i) > max_val) {
                max_val = xSeq.get(i);
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            xSeq.set(i, TornadoMath.exp(xSeq.get(i) - max_val));
            sum += xSeq.get(i);
        }
        // normalize
        for (int i = 0; i < size; i++) {
            xSeq.set(i, xSeq.get(i) / sum);
        }
    }

    // TODO: Similar to the RMSNorm, this version can be even slower than the sequential one due to type marshalling to get the streams to perform a parallel reduction.
    @Override
    public void computeWithJavaStreams() {
        // find max value (1for numerical stability)
        Optional<Float> max = Arrays //
                .stream(xStreams) //
                .parallel() //
                .max(Comparator.comparingDouble(Float::floatValue));
        float max_val = max.orElse(0.0f);

        // exp and sum
        IntStream.range(0, size)
                .parallel()
                .forEach(i -> xStreams[i] = TornadoMath.exp(xStreams[i] - max_val));

        final float sum = Arrays.stream(xStreams).reduce(0.0f, Float::sum);
        // normalization
        IntStream.range(0, size)
                .parallel()
                .forEach(i -> {
                    xStreams[i] = xStreams[i] / sum;
                });
        streams = true;
    }

    private void runThreads(Thread[] threads) throws InterruptedException {
        for (Thread t : threads) {
            t.start();
        }

        for (Thread t : threads) {
            t.join();
        }
    }

    @Override
    public void computeWithJavaThreads() throws InterruptedException {
        Range[] ranges = Utils.createRangesForCPU(x.getSize());
        final int maxProcessors = ranges.length;

        float[] buf = new float[size];
        for (int i = 0; i < size; i++) buf[i] = x.get(i);

        Thread[] threads = new Thread[maxProcessors];
        float[] reduction = new float[maxProcessors];

        // Phase 1: find max
        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                float maxValue = Float.MIN_VALUE;
                for (int j = ranges[t].min(); j < ranges[t].max(); j++) {
                    if (buf[j] > maxValue) maxValue = buf[j];
                }
                reduction[t] = maxValue;
            });
        });
        runThreads(threads);

        float max_value = Float.MIN_VALUE;
        for (float v : reduction) {
            if (v > max_value) max_value = v;
        }
        final float max_val = max_value;

        // Phase 2: exp(x - max)
        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                for (int j = ranges[t].min(); j < ranges[t].max(); j++) {
                    buf[j] = (float) Math.exp(buf[j] - max_val);
                }
            });
        });
        runThreads(threads);

        // Phase 3: sum
        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                float ss = 0.0f;
                for (int j = ranges[t].min(); j < ranges[t].max(); j++) {
                    ss += buf[j];
                }
                reduction[t] = ss;
            });
        });
        runThreads(threads);

        float ss = 0.0f;
        for (float v : reduction) ss += v;
        final float sum = ss;

        // Phase 4: normalize
        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                for (int j = ranges[t].min(); j < ranges[t].max(); j++) {
                    buf[j] = buf[j] / sum;
                }
            });
        });
        runThreads(threads);

        for (int i = 0; i < size; i++) x.set(i, buf[i]);
    }

    /**
     * Single threaded with Vector API.
     */
    @Override
    public void computeWithParallelVectorAPI() {
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        final int loopBound = species.loopBound(size);
        final long FLOAT_BYTES = 4;
        int i = 0;

        // Step 1: Find max (vectorized)
        float maxValue = Float.NEGATIVE_INFINITY;
        for (; i < loopBound; i += species.length()) {
            FloatVector vA = FloatVector.fromMemorySegment(species, x.getSegment(), i * FLOAT_BYTES, ByteOrder.nativeOrder());
            float blockMax = vA.reduceLanes(VectorOperators.MAX);
            if (blockMax > maxValue) maxValue = blockMax;
        }
        for (; i < size; i++) {
            if (x.get(i) > maxValue) maxValue = x.get(i);
        }

        // Step 2: exp(x - max) and sum — scalar, since Java Vector API has no vectorised exp
        i = 0;
        float sum = 0.0f;
        for (; i < size; i++) {
            float val = TornadoMath.exp(x.get(i) - maxValue);
            x.set(i, val);
            sum += val;
        }

        // Step 3: Normalize (vectorized)
        i = 0;
        for (; i < loopBound; i += species.length()) {
            FloatVector vX = FloatVector.fromMemorySegment(species, x.getSegment(), i * FLOAT_BYTES, ByteOrder.nativeOrder());
            FloatVector result = vX.div(sum);
            result.intoMemorySegment(x.getSegment(), i * FLOAT_BYTES, ByteOrder.nativeOrder());
        }
        for (; i < size; i++) {
            x.set(i, x.get(i) / sum);
        }
    }

    // ======================================================================
    // TornadoVM kernels — @Reduce pattern avoids local-barrier reductions
    // that fail on OpenCL CPU and Apple GPU backends.

    // Sequential scan: @Reduce max is unreliable on some GPU/OpenCL-CPU backends.
    // For n=1024 the sequential pass is negligible and always correct.
    private static void softmaxFindMax(FloatArray result, FloatArray x) {
        float max = x.get(0);
        for (int i = 1; i < x.getSize(); i++) {
            if (x.get(i) > max) max = x.get(i);
        }
        result.set(0, max);
    }

    private static void softmaxComputeExp(FloatArray x, FloatArray maxArr) {
        float max = maxArr.get(0);
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            x.set(i, TornadoMath.exp(x.get(i) - max));
        }
    }

    private static void softmaxComputeSum(@Reduce FloatArray result, FloatArray x) {
        result.set(0, 0.0f);
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            result.set(0, result.get(0) + x.get(i));
        }
    }

    private static void softmaxNormalize(FloatArray x, FloatArray sumArr) {
        float sum = sumArr.get(0);
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            x.set(i, x.get(i) / sum);
        }
    }

    // ======================================================================

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        TaskGraph taskGraph = new TaskGraph("benchmark")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x) //
                .task("softmax.findMax", SoftMax::softmaxFindMax, temp, x) //
                .task("softmax.computeExp", SoftMax::softmaxComputeExp, x, temp) //
                .task("softmax.computeSum", SoftMax::softmaxComputeSum, temp, x) //
                .task("softmax.normalize", SoftMax::softmaxNormalize, x, temp) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, x);

        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {

        private SoftMax rmsnorm;
        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            rmsnorm = new SoftMax(Catalog.DEFAULT.get(Catalog.BenchmarkID.SoftMax).size());
            executionPlan = rmsnorm.buildExecutionPlan();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void softMaxSequential(JMHBenchmark state) {
            state.rmsnorm.computeSequential();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void softMaxParallelStreams(JMHBenchmark state) {
            state.rmsnorm.computeWithJavaStreams();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void softMaxParallelThreads(JMHBenchmark state) {
            try {
                state.rmsnorm.computeWithJavaThreads();
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
        public void softMaxParallelVectorAPI(JMHBenchmark state) {
            state.rmsnorm.computeWithParallelVectorAPI();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void softMaxTornadoVM(JMHBenchmark state) {
            state.executionPlan.execute();
        }
    }

    @Override
    public void runWithJMH() throws RunnerException {
        org.openjdk.jmh.runner.options.Options opt = new OptionsBuilder() //
                .include(SoftMax.class.getName() + ".*") //
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
    public void resetOutputs() {
        streams = false;
        setInit();
    }

    private boolean validate(FloatArray outputRef, FloatArray output) {
        for (int i = 0; i < outputRef.getSize(); i++) {
            if (Math.abs(outputRef.get(i) - output.get(i)) > Config.DELTA) {
                System.out.println("ERROR: " + i + " != " + outputRef.get(i) + " vs " + output.get(i));
                return false;
            }
        }
        return true;
    }

    private boolean validate(FloatArray outputRef, Float[] output) {
        for (int i = 0; i < outputRef.getSize(); i++) {
            if (Math.abs(outputRef.get(i) - output[i]) > Config.DELTA) {
                System.out.println("ERROR: " + i + " != " + outputRef.get(i) + " vs " + output[i]);
                return false;
            }
        }
        return true;
    }


    @Override
    public void validate(int runID) {
        if (runID == 0) {
            if (streams) {
                System.out.println(" -- Result Correct? " + validate(xRef, xStreams));
            } else {
                System.out.println(" -- Result Correct? " + validate(xRef, x));
            }
        } else {
            System.out.println();
        }
        streams = false;
    }

    @Override
    public int getSize() {
        return size;
    }

    @Override
    public String getName() {
        return "softmax";
    }

    @Override
    public String printSize() {
        return getSize() + "";
    }

    public static void main(String[] args) throws InterruptedException {
        SoftMax benchmark = new SoftMax(Catalog.DEFAULT.get(Catalog.BenchmarkID.SoftMax).size());
        benchmark.run(args);
    }
}
