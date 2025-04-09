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
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
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
 *     tornado -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.SoftMax
 * </code>
 * </p>
 */
public class SoftMax extends BenchmarkDriver {

    private final int size;
    private FloatArray xRef;
    private FloatArray xInit;
    private FloatArray x;
    private FloatArray temp;
    private Float[] xStreams;
    boolean streams;

    public SoftMax(int size) {
        this.size = size;
        xRef = new FloatArray(size);
        xInit = new FloatArray(size);
        x = new FloatArray(size);
        temp = new FloatArray(size);
        xStreams = new Float[size];
        init();
        setInit();
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
        IntStream.range(0, size).forEach(i -> x.set(i, xInit.get(i)));
    }

    @Override
    public void computeSequential() {
        // find max value (for numerical stability)
        float max_val = xRef.get(0);
        for (int i = 1; i < size; i++) {
            if (xRef.get(i) > max_val) {
                max_val = xRef.get(i);
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            xRef.set(i, TornadoMath.exp(xRef.get(i) - max_val));
            sum += xRef.get(i);
        }
        // normalize
        for (int i = 0; i < size; i++) {
            xRef.set(i, xRef.get(i) / sum);
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
                .forEach(i -> xStreams[i] = TornadoMath.exp(xRef.get(i) - max_val));

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
        final int maxProcessors = Runtime.getRuntime().availableProcessors();

        Thread[] threads = new Thread[maxProcessors];
        // full reduction per thread
        float[] reduction = new float[maxProcessors];
        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                float maxValue = Float.MIN_VALUE;
                for (int j = ranges[t].min(); j < ranges[t].max(); j++) {
                    if (x.get(j) > maxValue) {
                        maxValue = x.get(j);
                    }
                }
                reduction[t] = maxValue;
            });
        });

        runThreads(threads);

        float max_value = Float.MIN_VALUE;
        for (float v : reduction) {
            if (v > max_value) {
                max_value = v;
            }
        }
        final float max_val = max_value;
        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                for (int j = ranges[t].min(); j < ranges[t].max(); j++) {
                    x.set(j, TornadoMath.exp(x.get(j) - max_val));
                }
            });
        });
        runThreads(threads);

        // Reduction
        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                float ss = 0.0f;
                for (int j = ranges[t].min(); j < ranges[t].max(); j++) {
                    ss += x.get(j);
                }
                reduction[t] = ss;
            });
        });

        runThreads(threads);

        // Sum reduction on CPU
        float ss = 0.0f;
        for (float v : reduction) {
            ss += v;
        }

        final float sum = ss;

        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                for (int j = ranges[t].min(); j < ranges[t].max(); j++) {
                    x.set(j , x.get(j) / sum);
                }
            });
        });

        runThreads(threads);
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
        float maxValue = 0.0f;
        for (; i < loopBound; i += species.length()) {
            FloatVector vA = FloatVector.fromMemorySegment(species, x.getSegment(), i * FLOAT_BYTES, ByteOrder.nativeOrder());
            maxValue = vA.mul(vA).reduceLanes(VectorOperators.MAX);
        }

        // The remaining part is done sequentially
        for (; i < size; i++) {
            if (x.get(i) > maxValue) {
                maxValue = x.get(i);
            }
        }
        i = 0;

        // exp and sum
        float sum = 0.0f;
        for (; i < loopBound; i += species.length()) {
            FloatVector vA = FloatVector.fromMemorySegment(species, x.getSegment(), i * FLOAT_BYTES, ByteOrder.nativeOrder());
            vA.intoMemorySegment(x.getSegment(), i * FLOAT_BYTES, ByteOrder.nativeOrder());
            sum += vA.reduceLanes(VectorOperators.ADD);
        }

        // The remaining part is done sequentially
        for (; i < size; i++) {
            sum += x.get(i);
        }

        i = 0;
        for (; i < loopBound; i += species.length()) {
            FloatVector vX = FloatVector.fromMemorySegment(species, x.getSegment(), i * FLOAT_BYTES, ByteOrder.nativeOrder());
            FloatVector result = vX.div(sum);
            result.intoMemorySegment(x.getSegment(), i * FLOAT_BYTES, ByteOrder.nativeOrder());
        }
        // The remaining part is done sequentially
        for (; i < size; i++) {
            x.set(i,  x.get(i) /  sum);
        }
    }

    private static void reductionOneBlock(KernelContext context, FloatArray output, FloatArray x) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupSize = context.localGroupSizeX;
        float[] localX = context.allocateFloatLocalArray(1024);
        localX[lid] = x.get(gid);
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] = TornadoMath.max(localX[lid], localX[lid + stride]);
            }
        }

        if (lid == 0) {
            // store max
            output.set(0, localX[0]);
        }
    }

    private static void reductionOneBlock2(KernelContext context, FloatArray output, FloatArray x) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupSize = context.localGroupSizeX;
        float[] localX = context.allocateFloatLocalArray(1024);
        localX[lid] = x.get(gid);
        float max = output.get(0);
        localX[lid] = TornadoMath.exp(localX[lid] - max);
        x.set(gid, localX[lid]);
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        if (lid == 0) {
            // final sum stored in ID 0
            output.set(0, localX[0]);
        }
    }

    private static void reductionOneBlock3(KernelContext context, FloatArray temp, FloatArray x) {
        int gid = context.globalIdx;
        float sum = temp.get(0);
        x.set(gid, x.get(gid) / sum);
    }

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        KernelContext kernelContext = new KernelContext();
        TaskGraph taskGraph = new TaskGraph("benchmark")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, x, temp) //
                .task("softmax.reduceMax", SoftMax::reductionOneBlock, kernelContext, temp, x) //
                .task("softmax.reduceExp", SoftMax::reductionOneBlock2, kernelContext, temp, x)  //
                .task("softmax.map", SoftMax::reductionOneBlock3, kernelContext, temp, x)  //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, x);

        WorkerGrid workerGrid = new WorkerGrid1D(size);
        workerGrid.setLocalWork(size, 1, 1);
        GridScheduler gridScheduler = new GridScheduler("benchmark.softmax.reduceMax", workerGrid);
        gridScheduler.addWorkerGrid("benchmark.softmax.reduceExp", workerGrid);
        gridScheduler.addWorkerGrid("benchmark.softmax.map", workerGrid);

        TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot());
        executionPlan.withGridScheduler(gridScheduler);
        return executionPlan;
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
    void runWithJMH() throws RunnerException {
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
                System.out.println(" -- Result Correct Streams? " + validate(xRef, xStreams));
            } else {
                System.out.println(" -- Result Correct? " + validate(xRef, x));
            }
        } else {
            System.out.println();
        }
        streams = false;
    }

    @Override
    int getSize() {
        return size;
    }

    @Override
    String getName() {
        return "RMSNorm";
    }

    @Override
    String printSize() {
        return getSize() + "";
    }

    public static void main(String[] args) throws InterruptedException {
        SoftMax benchmark = new SoftMax(Catalog.DEFAULT.get(Catalog.BenchmarkID.SoftMax).size());
        benchmark.run(args);
    }
}
