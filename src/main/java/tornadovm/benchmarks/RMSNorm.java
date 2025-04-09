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
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * Kernel taken from Llama2.cpp. This kernel is used as a first kernel in the chain
 * to perform LLM local inference. It contains a reduction and a map operation.
 *
 * <p>How to run?
 * <code>
 *     tornado -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.RMSNorm
 * </code>
 * </p>
 */
public class RMSNorm extends BenchmarkDriver {

    private final int size;
    private FloatArray outputRef;
    private FloatArray temp;
    private FloatArray output;
    private FloatArray x;
    private FloatArray weights;

    public RMSNorm(int size) {
        this.size = size;
        outputRef = new FloatArray(size);
        output = new FloatArray(size);
        temp = new FloatArray(size);
        x = new FloatArray(size);
        weights = new FloatArray(size);
        init();
    }

    private void init() {
        Random rand = new Random(71);
        for (int i = 0; i < size; i++) {
            x.set(i, rand.nextFloat());
            weights.set(i, rand.nextFloat());
        }
    }

    @Override
    public void computeSequential() {
        float ss = 0.0f;
        for (int i = 0; i < size; i++) {
            ss += x.get(i) * x.get(i);
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / TornadoMath.sqrt(ss);
        // normalize and scale
        for (int i = 0; i < size; i++) {
            outputRef.set(i,  weights.get(i) * (ss * x.get(i)));
        }
    }

    // TODO: Note that this version can be even slower than the sequential one due to type marshalling to get the streams to perform a parallel reduction.
    @Override
    public void computeWithJavaStreams() {
        Float[] temp = new Float[size];
        // Split the reduction into a map and then the reduction
        IntStream.range(0, size).parallel().forEach(i -> {
            temp[i] = x.get(i) * x.get(i);
        });

        // Reduction
        float ss = Arrays.stream(temp) //
                .parallel() //
                .reduce(0.0f, Float::sum);

        // normalize and scale
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / TornadoMath.sqrt(ss);
        final float ssFinal = ss;

        // final normalization (map operation)
        IntStream.range(0, size) //
                .parallel()      //
                .forEach(i -> { //
                    output.set(i, (weights.get(i) * (ssFinal * x.get(i)))); //
                });
    }

    @Override
    public void computeWithJavaThreads() throws InterruptedException {
        Range[] ranges = Utils.createRangesForCPU(output.getSize());
        final int maxProcessors = Runtime.getRuntime().availableProcessors();

        Thread[] threads = new Thread[maxProcessors];
        // full reduction per thread
        float[] reduction = new float[maxProcessors];
        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                float ss = 0.0f;
                for (int j = ranges[t].min(); j < ranges[t].max(); j++) {
                    ss += x.get(j) * x.get(j);
                }
                reduction[t] = ss;
            });
        });

        for (Thread t : threads) {
            t.start();
        }

        for (Thread t : threads) {
            t.join();
        }

        float ss = 0.0f;
        for (float v : reduction) {
            ss += v;
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / TornadoMath.sqrt(ss);

        final float ssFinal = ss;
        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                for (int j = ranges[t].min(); j < ranges[t].max(); j++) {
                    output.set(j, weights.get(j) * (ssFinal * x.get(j)));
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
     * Single threaded with Vector API.
     */
    @Override
    public void computeWithParallelVectorAPI() {
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        final int loopBound = species.loopBound(size);
        final long FLOAT_BYTES = 4;
        int i = 0;
        float ss = 0.0f;
        for (; i < loopBound; i += species.length()) {
            FloatVector vA = FloatVector.fromMemorySegment(species, x.getSegment(), i * FLOAT_BYTES, ByteOrder.nativeOrder());
            ss += vA.mul(vA).reduceLanes(VectorOperators.ADD);
        }

        // The remaining part is done sequentially
        for (; i < size; i++) {
            ss += x.get(i) * x.get(i);
        }

        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / TornadoMath.sqrt(ss);

        // normalize and scale
        i = 0;
        for (; i < loopBound; i += species.length()) {
            FloatVector vX = FloatVector.fromMemorySegment(species, x.getSegment(), i * FLOAT_BYTES, ByteOrder.nativeOrder());
            FloatVector vW = FloatVector.fromMemorySegment(species, weights.getSegment(), i * FLOAT_BYTES, ByteOrder.nativeOrder());
            FloatVector result = vW.mul(vX.mul(ss));
            result.intoMemorySegment(output.getSegment(), i * FLOAT_BYTES, ByteOrder.nativeOrder());
        }
        // The remaining part is done sequentially
        for (; i < size; i++) {
            output.set(i,  weights.get(i) * (ss * x.get(i)));
        }
    }

    // ======================================================================
    // TornadoVM Kernels
    private static void reduce(@Reduce FloatArray output, FloatArray x) {
        output.set(0, 0.0f);
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            float val = x.get(i) * x.get(i);
            output.set(0, output.get(0) + val);
        }
    }

    private static void singleNorm(FloatArray output, int size) {
        float ss = output.get(0);
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / TornadoMath.sqrt(ss);
        output.set(0, ss);
    }

    private static void map(FloatArray output, FloatArray weights, FloatArray x) {
        float acc = 1.0f;
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            output.set(i,  weights.get(i) * (acc * x.get(i)));
        }
    }


    /** Reductions launched in a single thread-block
     *
     * @param context
     * @param output
     * @param x
     * @param weights
     */
    private static void reductionOneBlock(KernelContext context, FloatArray output, FloatArray x, FloatArray weights) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupSize = context.localGroupSizeX;
        float[] localX = context.allocateFloatLocalArray(1024);
        localX[lid] = x.get(gid);
        localX[lid] = localX[lid] * localX[lid];
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        if (lid == 0) {
            float ss = localX[0];
            ss /= x.getSize();
            ss += 1e-5f;
            ss = 1.0f / TornadoMath.sqrt(ss);
            output.set(0, ss);
        }
    }

    private static void reductionOneBlock2(KernelContext context, FloatArray output, FloatArray x, FloatArray weights, FloatArray temp) {
        int gid = context.globalIdx;
        float ss = temp.get(0);
        output.set(gid, weights.get(gid) * (ss * x.get(gid)));
    }

    // ======================================================================

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        TaskGraph taskGraphLoop = new TaskGraph("benchmark")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights, x) //
                .task("reduce", RMSNorm::reduce, output, x) //
                .task("singleNorm", RMSNorm::singleNorm, output, size) //
                .task("map", RMSNorm::map, output, weights, x) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        KernelContext kernelContext = new KernelContext();
        TaskGraph taskGraph = new TaskGraph("benchmark")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, weights, x, temp) //
                .task("reductionsOneBlock", RMSNorm::reductionOneBlock, kernelContext, temp, x, weights) //
                .task("mapContext", RMSNorm::reductionOneBlock2, kernelContext, output, x, weights, temp)  //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, output);

        WorkerGrid workerGrid = new WorkerGrid1D(size);
        workerGrid.setLocalWork(size, 1, 1);
        GridScheduler gridScheduler = new GridScheduler("benchmark.reductionsOneBlock", workerGrid);
        gridScheduler.addWorkerGrid("benchmark.mapContext", workerGrid);

        TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot());
        executionPlan.withGridScheduler(gridScheduler);
        return executionPlan;
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {

        private RMSNorm rmsnorm;
        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            rmsnorm = new RMSNorm(Catalog.DEFAULT.get(Catalog.BenchmarkID.RMSNORM).size());
            executionPlan = rmsnorm.buildExecutionPlan();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void rmsNormSequential(JMHBenchmark state) {
            state.rmsnorm.computeSequential();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void rmsNormParallelStreams(JMHBenchmark state) {
            state.rmsnorm.computeWithJavaStreams();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void rmsNormParallelThreads(JMHBenchmark state) {
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
        public void rmsNormParallelVectorAPI(JMHBenchmark state) {
            state.rmsnorm.computeWithParallelVectorAPI();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void rmsNormTornadoVM(JMHBenchmark state) {
            state.executionPlan.execute();
        }
    }

    @Override
    void runWithJMH() throws RunnerException {
        org.openjdk.jmh.runner.options.Options opt = new OptionsBuilder() //
                .include(RMSNorm.class.getName() + ".*") //
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
        output.clear();
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

    @Override
    public void validate(int runID) {
        if (runID == 0) {
            System.out.println(" -- Result Correct? " + validate(outputRef, output));
        } else {
            System.out.println();
        }
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
        RMSNorm benchmark = new RMSNorm(Catalog.DEFAULT.get(Catalog.BenchmarkID.RMSNORM).size());
        benchmark.run(args);
    }
}
