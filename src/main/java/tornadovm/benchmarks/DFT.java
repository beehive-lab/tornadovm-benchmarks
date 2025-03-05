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
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.ByteOrder;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * How to run?
 * <code>
 *     tornado -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.DFT
 * </code>
 */
public class DFT extends BenchmarkDriver {

    private int size;
    private FloatArray inreal;
    private FloatArray inimag;
    private FloatArray outrealRef;
    private FloatArray outimagRef;
    private FloatArray outreal;
    private FloatArray outimag;

    public DFT(int size) {
        this.size = size;
        inreal = new FloatArray(size);
        inimag = new FloatArray(size);
        outreal = new FloatArray(size);
        outimag = new FloatArray(size);
        outrealRef = new FloatArray(size);
        outimagRef = new FloatArray(size);

        for (int i = 0; i < size; i++) {
            inreal.set(i, 1 / (float) (i + 2));
            inimag.set(i, 1 / (float) (i + 2));
        }
    }

    @Override
    public void computeSequential() {
        int n = inreal.getSize();
        for (int k = 0; k < n; k++) { // For each output element
            float sumreal = 0;
            float sumimag = 0;
            for (int t = 0; t < n; t++) { // For each input element
                float angle = ((2 * TornadoMath.floatPI() * t * k) / n);
                sumreal += (inreal.get(t) * (TornadoMath.cos(angle)) + inimag.get(t) * (TornadoMath.sin(angle)));
                sumimag += -(inreal.get(t) * (TornadoMath.sin(angle)) + inimag.get(t) * (TornadoMath.cos(angle)));
            }
            outrealRef.set(k, sumreal);
            outimagRef.set(k, sumimag);
        }
    }

    public void computeWithTornado(FloatArray inreal, FloatArray inimag, FloatArray outreal, FloatArray outimag) {
        int n = inreal.getSize();
        for (@Parallel int k = 0; k < n; k++) { // For each output element
            float sumreal = 0;
            float sumimag = 0;
            for (int t = 0; t < n; t++) { // For each input element
                float angle = ((2 * TornadoMath.floatPI() * t * k) / n);
                sumreal += (inreal.get(t) * (TornadoMath.cos(angle)) + inimag.get(t) * (TornadoMath.sin(angle)));
                sumimag += -(inreal.get(t) * (TornadoMath.sin(angle)) + inimag.get(t) * (TornadoMath.cos(angle)));
            }
            outreal.set(k, sumreal);
            outimag.set(k, sumimag);
        }
    }

    @Override
    public void computeWithJavaStreams() {
        int n = inreal.getSize();
        IntStream.range(0, n).parallel().forEach(k -> {
            float sumreal = 0;
            float sumimag = 0;
            for (int t = 0; t < n; t++) { // For each input element
                float angle = ((2 * TornadoMath.floatPI() * t * k) / n);
                sumreal += (inreal.get(t) * (TornadoMath.cos(angle)) + inimag.get(t) * (TornadoMath.sin(angle)));
                sumimag += -(inreal.get(t) * (TornadoMath.sin(angle)) + inimag.get(t) * (TornadoMath.cos(angle)));
            }
            outreal.set(k, sumreal);
            outimag.set(k, sumimag);
        });
    }

    @Override
    public void computeWithParallelVectorAPI() {
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
        final int FLOAT_BYTES = 4;
        int n = inreal.getSize();
        IntStream.range(0, n).parallel().forEach(k -> {
            float sumreal = 0;
            float sumimag = 0;
            for (int t = 0; t < n; t += species.length()) { // For each input element

                float[] angles = new float[species.length()];
                int tIndex = t;
                for (int i = 0; i < angles.length; i++) {
                    angles[i] = ((2 * TornadoMath.floatPI() * tIndex * k) / n);
                    tIndex++;
                }
                float[] cosAngles = new float[species.length()];
                for (int i = 0; i < cosAngles.length; i++) {
                    cosAngles[i] = (float) Math.cos(angles[i]);
                }

                float[] sinAngles = new float[species.length()];
                for (int i = 0; i < sinAngles.length; i++) {
                    sinAngles[i] = (float) Math.sin(angles[i]);
                }

                FloatVector vInReal = FloatVector.fromMemorySegment(species, inreal.getSegment(), t * FLOAT_BYTES, ByteOrder.nativeOrder());
                FloatVector vInImag = FloatVector.fromMemorySegment(species, inimag.getSegment(), t * FLOAT_BYTES, ByteOrder.nativeOrder());

                sumreal += vInReal.mul(FloatVector.fromArray(species, cosAngles, 0)).add(vInImag.mul(FloatVector.fromArray(species, sinAngles, 0))).reduceLanes(VectorOperators.ADD);
                sumimag += -1 * vInReal.mul(FloatVector.fromArray(species, sinAngles, 0)).add(vInImag.mul(FloatVector.fromArray(species, cosAngles, 0))).reduceLanes(VectorOperators.ADD);

            }
            outreal.set(k, sumreal);
            outimag.set(k, sumimag);
        });
    }

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        TaskGraph taskGraph = new TaskGraph("benchmark")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, inreal, inimag)
                .task("dft", this::computeWithTornado, inreal, inimag, outreal, outimag)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, outreal, outimag);
        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    @Override
    public void resetOutputs() {
        outimagRef.init(0);
        outrealRef.init(0);
    }

    @Override
    public void validate(int i) {
        validate(i, size, outrealRef, outimagRef, outreal, outimag);
    }

    @Override
    public void computeWithJavaThreads() throws InterruptedException {
        final int maxProcessors = Runtime.getRuntime().availableProcessors();
        Thread[] threads = new Thread[maxProcessors];
        int n = inreal.getSize();
        int balk = inreal.getSize() / maxProcessors;
        for (int current = 0; current < maxProcessors; current++) {
            int lowBound = current * balk;
            int upperBound = (current + 1) * balk;
            if (current == maxProcessors- 1) {
                upperBound = inreal.getSize();
            }
            int finalUpperBound = upperBound;
            threads[current] = new Thread(() -> {
                for (int k = lowBound; k < finalUpperBound; k++) {
                    float sumreal = 0;
                    float sumimag = 0;
                    for (int t = 0; t < inreal.getSize(); t++) { // For each input element
                        float angle = (float) ((2 * Math.PI * t * k) / (float) n);
                        sumreal += (float) (inreal.get(t) * (Math.cos(angle)) + inimag.get(t) * (Math.sin(angle)));
                        sumimag += -(float) (inreal.get(t) * (Math.sin(angle)) + inimag.get(t) * (Math.cos(angle)));
                    }
                    outreal.set(k, sumreal);
                    outimag.set(k, sumimag);
                }
            });
        }

        for (Thread t : threads) {
            t.start();
        }

        for (Thread t : threads) {
            t.join();
        }
    }

    public boolean validate(int size, FloatArray outRealRef, FloatArray outImagRef, FloatArray outReal, FloatArray outImag) {
        boolean val = true;
        for (int i = 0; i < size; i++) {
            if (Math.abs(outRealRef.get(i) - outReal.get(i)) > 1.0f) {
                System.out.println(outReal.get(i) + " vs " + outRealRef.get(i) + "\n");
                val = false;
                break;
            }
            if (Math.abs(outImagRef.get(i) - outImag.get(i)) > 1.0f) {
                System.out.println(outImagRef.get(i) + " vs " + outImag.get(i) + "\n");
                val = false;
                break;
            }
        }
        return val;
    }

    public void validate(int run, int size, FloatArray outRealRef, FloatArray outImaRef, FloatArray outReal, FloatArray outImag) {
        if (run == 0) {
            System.out.println(" -- Result Correct? " + validate(size, outRealRef, outImaRef, outReal, outImag));
        } else {
            System.out.println();
        }
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {

        private DFT dft;

        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            dft = new DFT(Catalog.DEFAULT.get(Catalog.BenchmarkID.DFT).size());
            executionPlan = dft.buildExecutionPlan();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void dftSequential(JMHBenchmark state) {
            state.dft.computeSequential();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void dftParallelStreams(JMHBenchmark state) {
            state.dft.computeWithJavaStreams();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void dftParallelThreads(JMHBenchmark state) {
            try {
                state.dft.computeWithJavaThreads();
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
        public void dftParallelVectorAPI(JMHBenchmark state) {
            state.dft.computeWithParallelVectorAPI();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void dftTornadoVM(JMHBenchmark state) {
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
                .include(DFT.class.getName() + ".*") //
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
        return "DFT";
    }

    @Override
    String printSize() {
        return "" + getSize();
    }

    public static void main(String[] args) throws InterruptedException {
        DFT benchmark = new DFT(Catalog.DEFAULT.get(Catalog.BenchmarkID.DFT).size());
        benchmark.run(args);
    }
}
