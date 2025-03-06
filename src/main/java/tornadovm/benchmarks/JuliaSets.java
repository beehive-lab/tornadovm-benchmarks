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
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class JuliaSets extends BenchmarkDriver {

    private static final int MAX_ITERATIONS = 1000;
    private final int size;
    private final FloatArray hue;
    private final FloatArray brightness;

    private final FloatArray hueRef;
    private final FloatArray brightnessRef;

    private static final float ZOOM = 1;
    private static final float CX = -0.7f;
    private static final float CY = 0.27015f;
    private static final float MOVE_X = 0;
    private static final float MOVE_Y = 0;

    public JuliaSets(int size) {
        this.size = size;
        hue = new FloatArray(size * size);
        brightness = new FloatArray(size * size);

        hueRef = new FloatArray(size * size);
        brightnessRef = new FloatArray(size * size);
    }

    @Override
    public void computeSequential() {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float zx = 1.5f * (i - size / 2) / (0.5f * ZOOM * size) + MOVE_X;
                float zy = (j - size / 2) / (0.5f * ZOOM * size) + MOVE_Y;
                float k = MAX_ITERATIONS;
                while (zx * zx + zy * zy < 4 && k > 0) {
                    float tmp = zx * zx - zy * zy + CX;
                    zy = 2.0f * zx * zy + CY;
                    zx = tmp;
                    k--;
                }
                hueRef.set(i * size + j, (MAX_ITERATIONS / k));
                brightnessRef.set(i * size + j, k > 0 ? 1 : 0);
            }
        }
    }

    @Override
    public void computeWithJavaStreams() {
        IntStream.range(0, size).parallel().forEach(i -> {
            IntStream.range(0, size).parallel().forEach(j -> {
                float zx = 1.5f * (i - size / 2) / (0.5f * ZOOM * size) + MOVE_X;
                float zy = (j - size / 2) / (0.5f * ZOOM * size) + MOVE_Y;
                float k = MAX_ITERATIONS;
                while (zx * zx + zy * zy < 4 && k > 0) {
                    float tmp = zx * zx - zy * zy + CX;
                    zy = 2.0f * zx * zy + CY;
                    zx = tmp;
                    k--;
                }
                hue.set(i * size + j, (MAX_ITERATIONS / k));
                brightness.set(i * size + j, k > 0 ? 1 : 0);
            });
        });
    }

    @Override
    public void computeWithJavaThreads() throws InterruptedException {
        Range[] ranges = Utils.createRangesForCPU(size);
        int maxProcessors = Runtime.getRuntime().availableProcessors();
        Thread[] threads = new Thread[maxProcessors];
        IntStream.range(0, threads.length).forEach(threadIndex -> {
            threads[threadIndex] = new Thread(() -> {
                for (int i = ranges[threadIndex].min(); i < ranges[threadIndex].max(); i++) {
                    for (int j = 0; j < size; j++) {
                        float zx = 1.5f * (i - size / 2) / (0.5f * ZOOM * size) + MOVE_X;
                        float zy = (j - size / 2) / (0.5f * ZOOM * size) + MOVE_Y;
                        float k = MAX_ITERATIONS;
                        while (zx * zx + zy * zy < 4 && k > 0) {
                            float tmp = zx * zx - zy * zy + CX;
                            zy = 2.0f * zx * zy + CY;
                            zx = tmp;
                            k--;
                        }
                        hue.set(i * size + j, (MAX_ITERATIONS / k));
                        brightness.set(i * size + j, k > 0 ? 1 : 0);
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
        throw new UnsupportedOperationException("Not supported yet.");
    }

    private void computeWithTornadoVM(int size, FloatArray hue, FloatArray brightness) {
        for (@Parallel int i = 0; i < size; i++) {
            for (@Parallel int j = 0; j < size; j++) {
                float zx = 1.5f * (i - size / 2) / (0.5f * ZOOM * size) + MOVE_X;
                float zy = (j - size / 2) / (0.5f * ZOOM * size) + MOVE_Y;
                float k = MAX_ITERATIONS;
                while (zx * zx + zy * zy < 4 && k > 0) {
                    float tmp = zx * zx - zy * zy + CX;
                    zy = 2.0f * zx * zy + CY;
                    zx = tmp;
                    k--;
                }
                hue.set(i * size + j, (MAX_ITERATIONS / k));
                brightness.set(i * size + j, k > 0 ? 1 : 0);
            }
        }
    }

        @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        TaskGraph taskGraph = new TaskGraph("benchmark") //
                .task("juliaSet", this::computeWithTornadoVM, size, hue, brightness) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, hue, brightness);
        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    @Override
    public void resetOutputs() {
        hue.clear();
        brightness.clear();
    }

    private boolean validate(int numBodies, FloatArray hue, FloatArray brightness, FloatArray hueRef, FloatArray brightnessRef) {
        for (int i = 0; i < numBodies * 4; i++) {
            if (Math.abs(hue.get(i) - hueRef.get(i)) > Config.DELTA) {
                return false;
            }
            if (Math.abs(brightness.get(i) - brightnessRef.get(i)) > Config.DELTA) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void validate(int run) {
        if (run == 0) {
            System.out.println(" -- Result Correct? " + validate(size, hue, brightness, brightnessRef, brightnessRef));
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
        private JuliaSets juliaSet;
        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            juliaSet = new JuliaSets(Catalog.DEFAULT.get(Catalog.BenchmarkID.NBody).size());
            executionPlan = juliaSet.buildExecutionPlan();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void juliasetSequential(JMHBenchmark state) {
            state.juliaSet.computeSequential();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void juliasetParallelStreams(JMHBenchmark state) {
            state.juliaSet.computeWithJavaStreams();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void juliasetParallelThreads(JMHBenchmark state) throws InterruptedException {
            state.juliaSet.computeWithJavaThreads();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void juliasetParallelVectorAPI(JMHBenchmark state) {
            state.juliaSet.computeWithParallelVectorAPI();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void juliasetTornadoVM(JMHBenchmark state) {
            state.executionPlan.execute();
        }
    }

    @Override
    void runWithJMH() throws RunnerException {
        org.openjdk.jmh.runner.options.Options opt = new OptionsBuilder() //
                .include(JuliaSets.class.getName() + ".*") //
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
        return "juliasets";
    }

    @Override
    String printSize() {
        return getSize() + "x" + getSize();
    }

    public static void main(String[] args) throws InterruptedException {
        JuliaSets benchmark = new JuliaSets(Catalog.DEFAULT.get(Catalog.BenchmarkID.JuliaSets).size());
        benchmark.run(args);
    }
}
