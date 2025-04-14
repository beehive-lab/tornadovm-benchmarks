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

import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class Blackscholes extends BenchmarkDriver {

    private int size;
    private FloatArray input;
    private FloatArray callResultRef;
    private FloatArray putResultRef;
    private FloatArray callResult;
    private FloatArray putResult;

    public Blackscholes(int size) {
        this.size = size;
        input = new FloatArray(size);
        callResult = new FloatArray(size);
        callResultRef = new FloatArray(size);
        putResult = new FloatArray(size);
        putResultRef = new FloatArray(size);
        Random r = new Random(71);
        for (int i = 0; i < size; i++) {
            input.set(i, r.nextFloat());
        }
    }

    private static float cnd(float X) {
        final float c1 = 0.319381530f;
        final float c2 = -0.356563782f;
        final float c3 = 1.781477937f;
        final float c4 = -1.821255978f;
        final float c5 = 1.330274429f;
        final float zero = 0.0f;
        final float one = 1.0f;
        final float two = 2.0f;
        final float temp4 = 0.2316419f;
        final float oneBySqrt2pi = 0.398942280f;
        float absX = TornadoMath.abs(X);
        float t = one / (one + temp4 * absX);
        float y = one - oneBySqrt2pi * TornadoMath.exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
        return ((X < zero) ? (one - y) : y);
    }

    @Override
    public void computeSequential() {
        for (int idx = 0; idx < callResultRef.getSize(); idx++) {
            float rand = input.get(idx);
            final float S_LOWER_LIMIT = 10.0f;
            final float S_UPPER_LIMIT = 100.0f;
            final float K_LOWER_LIMIT = 10.0f;
            final float K_UPPER_LIMIT = 100.0f;
            final float T_LOWER_LIMIT = 1.0f;
            final float T_UPPER_LIMIT = 10.0f;
            final float R_LOWER_LIMIT = 0.01f;
            final float R_UPPER_LIMIT = 0.05f;
            final float SIGMA_LOWER_LIMIT = 0.01f;
            final float SIGMA_UPPER_LIMIT = 0.10f;
            final float S = S_LOWER_LIMIT * rand + S_UPPER_LIMIT * (1.0f - rand);
            final float K = K_LOWER_LIMIT * rand + K_UPPER_LIMIT * (1.0f - rand);
            final float T = T_LOWER_LIMIT * rand + T_UPPER_LIMIT * (1.0f - rand);
            final float r = R_LOWER_LIMIT * rand + R_UPPER_LIMIT * (1.0f - rand);
            final float v = SIGMA_LOWER_LIMIT * rand + SIGMA_UPPER_LIMIT * (1.0f - rand);

            float d1 = (TornadoMath.log(S / K) + ((r + (v * v / 2)) * T)) / v * TornadoMath.sqrt(T);
            float d2 = d1 - (v * TornadoMath.sqrt(T));
            callResultRef.set(idx, S * cnd(d1) - K * TornadoMath.exp(T * (-1) * r) * cnd(d2));
            putResultRef.set(idx, K * TornadoMath.exp(T * -r) * cnd(-d2) - S * cnd(-d1));
        }
    }

    @Override
    public void computeWithJavaStreams() {
        IntStream.range(0, callResult.getSize()).parallel().forEach(idx -> {
            float rand = input.get(idx);
            final float S_LOWER_LIMIT = 10.0f;
            final float S_UPPER_LIMIT = 100.0f;
            final float K_LOWER_LIMIT = 10.0f;
            final float K_UPPER_LIMIT = 100.0f;
            final float T_LOWER_LIMIT = 1.0f;
            final float T_UPPER_LIMIT = 10.0f;
            final float R_LOWER_LIMIT = 0.01f;
            final float R_UPPER_LIMIT = 0.05f;
            final float SIGMA_LOWER_LIMIT = 0.01f;
            final float SIGMA_UPPER_LIMIT = 0.10f;
            final float S = S_LOWER_LIMIT * rand + S_UPPER_LIMIT * (1.0f - rand);
            final float K = K_LOWER_LIMIT * rand + K_UPPER_LIMIT * (1.0f - rand);
            final float T = T_LOWER_LIMIT * rand + T_UPPER_LIMIT * (1.0f - rand);
            final float r = R_LOWER_LIMIT * rand + R_UPPER_LIMIT * (1.0f - rand);
            final float v = SIGMA_LOWER_LIMIT * rand + SIGMA_UPPER_LIMIT * (1.0f - rand);

            float d1 = (TornadoMath.log(S / K) + ((r + (v * v / 2)) * T)) / v * TornadoMath.sqrt(T);
            float d2 = d1 - (v * TornadoMath.sqrt(T));
            callResult.set(idx, S * cnd(d1) - K * TornadoMath.exp(T * (-1) * r) * cnd(d2));
            putResult.set(idx, K * TornadoMath.exp(T * -r) * cnd(-d2) - S * cnd(-d1));
        });

    }

    @Override
    public void computeWithJavaThreads() throws InterruptedException {
        int maxProcessors = Runtime.getRuntime().availableProcessors();
        Range[] ranges = Utils.createRangesForCPU(size);
        Thread[] threads = new Thread[maxProcessors];
        IntStream.range(0, threads.length).forEach(t -> {
            threads[t] = new Thread(() -> {
                for (int i = ranges[t].min(); i < ranges[t].max(); i++) {
                    float rand = input.get(i);
                    final float S_LOWER_LIMIT = 10.0f;
                    final float S_UPPER_LIMIT = 100.0f;
                    final float K_LOWER_LIMIT = 10.0f;
                    final float K_UPPER_LIMIT = 100.0f;
                    final float T_LOWER_LIMIT = 1.0f;
                    final float T_UPPER_LIMIT = 10.0f;
                    final float R_LOWER_LIMIT = 0.01f;
                    final float R_UPPER_LIMIT = 0.05f;
                    final float SIGMA_LOWER_LIMIT = 0.01f;
                    final float SIGMA_UPPER_LIMIT = 0.10f;
                    final float S = S_LOWER_LIMIT * rand + S_UPPER_LIMIT * (1.0f - rand);
                    final float K = K_LOWER_LIMIT * rand + K_UPPER_LIMIT * (1.0f - rand);
                    final float T = T_LOWER_LIMIT * rand + T_UPPER_LIMIT * (1.0f - rand);
                    final float r = R_LOWER_LIMIT * rand + R_UPPER_LIMIT * (1.0f - rand);
                    final float v = SIGMA_LOWER_LIMIT * rand + SIGMA_UPPER_LIMIT * (1.0f - rand);

                    float d1 = (TornadoMath.log(S / K) + ((r + (v * v / 2)) * T)) / v * TornadoMath.sqrt(T);
                    float d2 = d1 - (v * TornadoMath.sqrt(T));
                    callResult.set(i, S * cnd(d1) - K * TornadoMath.exp(T * (-1) * r) * cnd(d2));
                    putResult.set(i, K * TornadoMath.exp(T * -r) * cnd(-d2) - S * cnd(-d1));
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

    final float S_LOWER_LIMIT = 10.0f;
    final float S_UPPER_LIMIT = 100.0f;
    final float K_LOWER_LIMIT = 10.0f;
    final float K_UPPER_LIMIT = 100.0f;
    final float T_LOWER_LIMIT = 1.0f;
    final float T_UPPER_LIMIT = 10.0f;
    final float R_LOWER_LIMIT = 0.01f;
    final float R_UPPER_LIMIT = 0.05f;
    final float SIGMA_LOWER_LIMIT = 0.01f;
    final float SIGMA_UPPER_LIMIT = 0.10f;
    @Override
    public void computeWithParallelVectorAPI() {
        for (int idx = 0; idx < callResultRef.getSize(); idx++) {
            float val = input.get(idx);
            final float S = S_LOWER_LIMIT * val + S_UPPER_LIMIT * (1.0f - val);
            final float K = K_LOWER_LIMIT * val + K_UPPER_LIMIT * (1.0f - val);
            final float T = T_LOWER_LIMIT * val + T_UPPER_LIMIT * (1.0f - val);
            final float r = R_LOWER_LIMIT * val + R_UPPER_LIMIT * (1.0f - val);
            final float v = SIGMA_LOWER_LIMIT * val + SIGMA_UPPER_LIMIT * (1.0f - val);

            float d1 = (TornadoMath.log(S / K) + ((r + (v * v / 2)) * T)) / v * TornadoMath.sqrt(T);
            float d2 = d1 - (v * TornadoMath.sqrt(T));
            callResultRef.set(idx, S * cnd(d1) - K * TornadoMath.exp(T * (-1) * r) * cnd(d2));
            putResultRef.set(idx, K * TornadoMath.exp(T * -r) * cnd(-d2) - S * cnd(-d1));
        }
    }

    private static void blackScholesKernel(FloatArray input, FloatArray callResult, FloatArray putResult) {
        for (@Parallel int idx = 0; idx < callResult.getSize(); idx++) {
            float rand = input.get(idx);
            final float S_LOWER_LIMIT = 10.0f;
            final float S_UPPER_LIMIT = 100.0f;
            final float K_LOWER_LIMIT = 10.0f;
            final float K_UPPER_LIMIT = 100.0f;
            final float T_LOWER_LIMIT = 1.0f;
            final float T_UPPER_LIMIT = 10.0f;
            final float R_LOWER_LIMIT = 0.01f;
            final float R_UPPER_LIMIT = 0.05f;
            final float SIGMA_LOWER_LIMIT = 0.01f;
            final float SIGMA_UPPER_LIMIT = 0.10f;
            final float S = S_LOWER_LIMIT * rand + S_UPPER_LIMIT * (1.0f - rand);
            final float K = K_LOWER_LIMIT * rand + K_UPPER_LIMIT * (1.0f - rand);
            final float T = T_LOWER_LIMIT * rand + T_UPPER_LIMIT * (1.0f - rand);
            final float r = R_LOWER_LIMIT * rand + R_UPPER_LIMIT * (1.0f - rand);
            final float v = SIGMA_LOWER_LIMIT * rand + SIGMA_UPPER_LIMIT * (1.0f - rand);

            float d1 = (TornadoMath.log(S / K) + ((r + (v * v / 2)) * T)) / v * TornadoMath.sqrt(T);
            float d2 = d1 - (v * TornadoMath.sqrt(T));
            callResult.set(idx, S * cnd(d1) - K * TornadoMath.exp(T * (-1) * r) * cnd(d2));
            putResult.set(idx, K * TornadoMath.exp(T * -r) * cnd(-d2) - S * cnd(-d1));
        }
    }

    @Override
    public TornadoExecutionPlan buildExecutionPlan()  {
        TaskGraph taskGraph = new TaskGraph("bechmark") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, input) //
                .task("blackscholes", Blackscholes::blackScholesKernel, input, callResult, putResult) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, callResult, putResult);
        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    @Override
    public void resetOutputs() {
        callResult.init(0);
        putResult.init(0);
    }

    private static boolean validate(FloatArray callRef, FloatArray putRef, FloatArray callPrice, FloatArray putPrice) {
        double delta = 1.8;
        for (int i = 0; i < callRef.getSize(); i++) {
            if (Math.abs(callRef.get(i) - callPrice.get(i)) > delta) {
                System.out.println("call: " + callRef.get(i) + " vs gpu " + callPrice.get(i));
                return false;
            }
            if (Math.abs(putRef.get(i) - putPrice.get(i)) > delta) {
                System.out.println("put: " + putRef.get(i) + " vs gpu " + putPrice.get(i));
                return false;
            }
        }
        return true;
    }

    @Override
    public void validate(int run) {
        if (run == 0) {
            System.out.println(" -- Result Correct? " + validate(callResultRef, putResultRef, callResult, putResult));
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

        private Blackscholes blackscholes;
        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            blackscholes = new Blackscholes(Catalog.DEFAULT.get(Catalog.BenchmarkID.Blackscholes).size());
            executionPlan = blackscholes.buildExecutionPlan();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void blackscholesSequential(JMHBenchmark state) {
            state.blackscholes.computeSequential();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void blackscholesParallelStreams(JMHBenchmark state) {
            state.blackscholes.computeWithJavaStreams();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void blackscholesParallelThreads(JMHBenchmark state) {
            try {
                state.blackscholes.computeWithJavaThreads();
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
        public void blackscholesParallelVectorAPI(JMHBenchmark state) {
            state.blackscholes.computeWithParallelVectorAPI();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void blackscholesTornadoVM(JMHBenchmark state) {
            state.executionPlan.execute();
        }
    }

    @Override
    void runWithJMH() throws RunnerException {
        org.openjdk.jmh.runner.options.Options opt = new OptionsBuilder() //
                .include(Blackscholes.class.getName() + ".*") //
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
        return "blackcholes";
    }

    @Override
    String printSize() {
        return "" + getSize();
    }

    public static void main(String[] args) throws InterruptedException {
        Blackscholes benchmark = new Blackscholes(Catalog.DEFAULT.get(Catalog.BenchmarkID.Blackscholes).size());
        benchmark.run(args);
    }
}
