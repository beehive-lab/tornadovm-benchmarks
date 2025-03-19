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

import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * Benchmark taken from the TornadoVM internal suite.
 *
 * <p>
 *     How to run it?
 *     <code>
 *         java -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.Montecarlo
 *     </code>
 * </p>
 */
public class Montecarlo extends BenchmarkDriver {

    int size;

    FloatArray outputRef;
    FloatArray output;
    int iterations;

    public Montecarlo(int size) {
        this.size = size;
        outputRef = new FloatArray(size);
        output = new FloatArray(size);
        iterations = size;
    }

    @Override
    public void computeSequential() {
        for (@Parallel int j = 0; j < iterations; j++) {
            long seed = j;
            // generate a pseudo random number (you do need it twice)
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);

            // this generates a number between 0 and 1 (with an awful entropy)
            float x = (seed & 0x0FFFFFFF) / 268435455f;

            // repeat for y
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            float y = (seed & 0x0FFFFFFF) / 268435455f;

            float dist = (float) Math.sqrt(x * x + y * y);
            if (dist <= 1.0f) {
                outputRef.set(j, 1.0f);
            } else {
                outputRef.set(j, 0.0f);
            }
        }
    }

    @Override
    public void computeWithJavaStreams() {
        IntStream.range(0, iterations).parallel().forEach(j -> {
            long seed = j;
            // generate a pseudo random number (you do need it twice)
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);

            // this generates a number between 0 and 1 (with an awful entropy)
            float x = (seed & 0x0FFFFFFF) / 268435455f;

            // repeat for y
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            float y = (seed & 0x0FFFFFFF) / 268435455f;

            float dist = (float) Math.sqrt(x * x + y * y);
            if (dist <= 1.0f) {
                output.set(j, 1.0f);
            } else {
                output.set(j, 0.0f);
            }
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
                    long seed = j;
                    // generate a pseudo random number (you do need it twice)
                    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);

                    // this generates a number between 0 and 1 (with an awful entropy)
                    float x = (seed & 0x0FFFFFFF) / 268435455f;

                    // repeat for y
                    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                    float y = (seed & 0x0FFFFFFF) / 268435455f;

                    float dist = (float) Math.sqrt(x * x + y * y);
                    if (dist <= 1.0f) {
                        output.set(j, 1.0f);
                    } else {
                        output.set(j, 0.0f);
                    }
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
        for (int j = 0; j < iterations; j+= species.length()) {

            // Create vector x and vector y
            float[] x = new float[species.length()];
            float[] y = new float[species.length()];
            for (int i = 0; i < species.length(); i++) {
                long seed = j + i;
                // generate a pseudo random number (you do need it twice)
                seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);

                // this generates a number between 0 and 1 (with an awful entropy)
                x[i] = (seed & 0x0FFFFFFF) / 268435455f;

                // repeat for y
                seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
                y[i] = (seed & 0x0FFFFFFF) / 268435455f;
            }

            FloatVector vX = FloatVector.fromArray(species, x, 0);
            FloatVector vY = FloatVector.fromArray(species, y, 0);
            FloatVector mulX = vX.mul(vX);
            FloatVector mulY = vY.mul(vY);
            float[] dist = mulX.add(mulY).sqrt().toArray();

            for (float v : dist) {
                if (v <= 1.0f) {
                    output.set(j, 1.0f);
                } else {
                    output.set(j, 0.0f);
                }
            }
        }
    }

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        TaskGraph taskGraph = new TaskGraph("benchmark")
            .task("montecarlo", Montecarlo::computeWithTornadoVM, output, size)
            .transferToHost(DataTransferMode.EVERY_EXECUTION, output);
        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    @Override
    public void resetOutputs() {
        output.init(0);
    }

    @Override
    public void validate(int i) {
        validate(i, outputRef, output);
    }

    private static void computeWithTornadoVM(FloatArray output, final int iterations) {
        for (@Parallel int j = 0; j < iterations; j++) {
            long seed = j;
            // generate a pseudo random number (you do need it twice)
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);

            // this generates a number between 0 and 1 (with an awful entropy)
            float x = (seed & 0x0FFFFFFF) / 268435455f;

            // repeat for y
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
            float y = (seed & 0x0FFFFFFF) / 268435455f;

            float dist = (float) Math.sqrt(x * x + y * y);
            if (dist <= 1.0f) {
                output.set(j, 1.0f);
            } else {
                output.set(j, 0.0f);
            }
        }
    }

    private boolean validate(FloatArray outputRef, FloatArray output) {
        for (int i = 0; i < outputRef.getSize(); i++) {
            if (Math.abs(outputRef.get(i) - output.get(i)) > 0.1) {
                System.out.println(outputRef.get(i) + " != " + output.get(i));
                return false;
            }
        }
        return true;
    }

    private void validate(int run, FloatArray outputRef, FloatArray output) {
        if (run == 0) {
            System.out.println(" -- Result Correct? " + validate(outputRef, output));
        } else {
            System.out.println();
        }
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {
        private Montecarlo montecarlo;
        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            montecarlo = new Montecarlo(Catalog.DEFAULT.get(Catalog.BenchmarkID.Montecarlo).size());
            executionPlan = montecarlo.buildExecutionPlan();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void montecarloSequential(JMHBenchmark state) {
            state.montecarlo.computeSequential();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void montecarloParallelStreams(JMHBenchmark state) {
            state.montecarlo.computeWithJavaStreams();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void montecarloParallelThreads(JMHBenchmark state) throws InterruptedException {
            state.montecarlo.computeWithJavaThreads();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void montecarloParallelVectorAPI(JMHBenchmark state) {
            state.montecarlo.computeWithParallelVectorAPI();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void montecarloTornadoVM(JMHBenchmark state) {
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
                .include(Montecarlo.class.getName() + ".*") //
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
        return "montecarlo";
    }

    @Override
    String printSize() {
        return "" + getSize();
    }

    public static void main(String[] args) throws InterruptedException {
        Montecarlo benchmark = new Montecarlo(Catalog.DEFAULT.get(Catalog.BenchmarkID.Montecarlo).size());
        benchmark.run(args);
    }
}
