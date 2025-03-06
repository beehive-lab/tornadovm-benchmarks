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

import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class NBody extends BenchmarkDriver {

    private final int numBodies;
    private float delT;
    private float espSqr;
    private FloatArray posRef;
    private FloatArray velRef;
    private FloatArray pos;
    private FloatArray vel;

    FloatArray auxPositionRandom;
    FloatArray auxVelocityZero;

    public NBody(int numBodies) {
        this.delT = 0.005f;
        this.espSqr = 500.0f;
        this.numBodies = numBodies;
        auxPositionRandom = new FloatArray(numBodies * 4);
        auxVelocityZero = new FloatArray(numBodies * 3);
        for (int i = 0; i < auxPositionRandom.getSize(); i++) {
            auxPositionRandom.set(i, (float) Math.random());
        }
        auxVelocityZero.init(0.0f);
        posRef = new FloatArray(numBodies * 4);
        velRef = new FloatArray(numBodies * 4);

        pos = new FloatArray(numBodies * 4);
        vel = new FloatArray(numBodies * 4);
        init();
    }

    private void init() {
        IntStream.range(0, auxPositionRandom.getSize()).forEach(i -> {
            posRef.set(i, auxPositionRandom.get(i));
            pos.set(i, auxPositionRandom.get(i));
        });
        IntStream.range(0, auxVelocityZero.getSize()).forEach(i -> {
            velRef.set(i, auxVelocityZero.get(i));
            vel.set(i, auxVelocityZero.get(i));
        });
    }

    @Override
    public void computeSequential() {
        for (int i = 0; i < numBodies; i++) {
            int body = 4 * i;
            float[] acc = new float[] { 0.0f, 0.0f, 0.0f };
            for (int j = 0; j < numBodies; j++) {
                float[] r = new float[3];
                int index = 4 * j;

                float distSqr = 0.0f;
                for (int k = 0; k < 3; k++) {
                    r[k] = pos.get(index + k) - pos.get(body + k);
                    distSqr += r[k] * r[k];
                }

                float invDist = 1.0f / TornadoMath.sqrt(distSqr + espSqr);

                float invDistCube = invDist * invDist * invDist;
                float s = pos.get(index + 3) * invDistCube;

                for (int k = 0; k < 3; k++) {
                    acc[k] += s * r[k];
                }
            }
            for (int k = 0; k < 3; k++) {
                pos.set(body + k, pos.get(body + k) + pos.get(body + k) * delT + 0.5f * acc[k] * delT * delT);
                vel.set(body + k, pos.get(body + k) + acc[k] * delT);
            }
        }
    }

    @Override
    public void computeWithJavaStreams() {
        IntStream.range(0, numBodies).parallel().forEach(bodyID -> {
            int body = 4 * bodyID;
            float[] acc = new float[]{0.0f, 0.0f, 0.0f};
            for (int j = 0; j < numBodies; j++) {
                float[] r = new float[3];
                int index = 4 * j;

                float distSqr = 0.0f;
                for (int k = 0; k < 3; k++) {
                    r[k] = pos.get(index + k) - pos.get(body + k);
                    distSqr += r[k] * r[k];
                }

                float invDist = 1.0f / TornadoMath.sqrt(distSqr + espSqr);

                float invDistCube = invDist * invDist * invDist;
                float s = pos.get(index + 3) * invDistCube;

                for (int k = 0; k < 3; k++) {
                    acc[k] += s * r[k];
                }
            }
            // Update x,y,z positions and velocity
            for (int k = 0; k < 3; k++) {
                pos.set(body + k, pos.get(body + k) + pos.get(body + k) * delT + 0.5f * acc[k] * delT * delT);
                vel.set(body + k, pos.get(body + k) + acc[k] * delT);
            }
        });
    }

    @Override
    public void computeWithJavaThreads() throws InterruptedException {

        Range[] ranges = Utils.createRangesForCPU(numBodies);
        int maxProcessors = Runtime.getRuntime().availableProcessors();
        Thread[] threads = new Thread[maxProcessors];
        IntStream.range(0, threads.length).forEach(threadIndex -> {
            threads[threadIndex] = new Thread(() -> {
                for (int bodyID = ranges[threadIndex].min(); bodyID < ranges[threadIndex].max(); bodyID++) {
                    int body = 4 * bodyID;
                    float[] acc = new float[]{0.0f, 0.0f, 0.0f};
                    for (int j = 0; j < numBodies; j++) {
                        float[] r = new float[3];
                        int index = 4 * j;

                        float distSqr = 0.0f;
                        for (int k = 0; k < 3; k++) {
                            r[k] = posRef.get(index + k) - posRef.get(body + k);
                            distSqr += r[k] * r[k];
                        }

                        float invDist = 1.0f / TornadoMath.sqrt(distSqr + espSqr);

                        float invDistCube = invDist * invDist * invDist;
                        float s = posRef.get(index + 3) * invDistCube;

                        for (int k = 0; k < 3; k++) {
                            acc[k] += s * r[k];
                        }
                    }
                    for (int k = 0; k < 3; k++) {
                        posRef.set(body + k, posRef.get(body + k) + posRef.get(body + k) * delT + 0.5f * acc[k] * delT * delT);
                        velRef.set(body + k, posRef.get(body + k) + acc[k] * delT);
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

    public void computeWithTornadoVM(int numBodies, FloatArray refPos, FloatArray refVel, float delT, float espSqr) {
        for (@Parallel int i = 0; i < numBodies; i++) {
            int body = 4 * i;
            float[] acc = new float[] { 0.0f, 0.0f, 0.0f };
            for (int j = 0; j < numBodies; j++) {
                float[] r = new float[3];
                int index = 4 * j;

                float distSqr = 0.0f;
                for (int k = 0; k < 3; k++) {
                    r[k] = refPos.get(index + k) - refPos.get(body + k);
                    distSqr += r[k] * r[k];
                }

                float invDist = 1.0f / TornadoMath.sqrt(distSqr + espSqr);

                float invDistCube = invDist * invDist * invDist;
                float s = refPos.get(index + 3) * invDistCube;

                for (int k = 0; k < 3; k++) {
                    acc[k] += s * r[k];
                }
            }
            for (int k = 0; k < 3; k++) {
                refPos.set(body + k, refPos.get(body + k) + refPos.get(body + k) * delT + 0.5f * acc[k] * delT * delT);
                refVel.set(body + k, refPos.get(body + k) + acc[k] * delT);
            }
        }
    }

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        TaskGraph taskGraph = new TaskGraph("benchmark")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, vel, pos) //
                .task("nbody", this::computeWithTornadoVM, numBodies, pos, vel, delT, espSqr)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, vel, pos);
        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    @Override
    public void resetOutputs() {
        // This application is iterative, every time we run we obtain
        // a new position and a new velocity. To compare, we set
        // the initial position and velocity.
        init();
    }

    private boolean validate(int numBodies, FloatArray pos, FloatArray vel, FloatArray posRef, FloatArray velRef) {
        for (int i = 0; i < numBodies * 4; i++) {
            if (Math.abs(vel.get(i) - velRef.get(i)) > Config.DELTA) {
                return false;
            }
            if (Math.abs(pos.get(i) - posRef.get(i)) > Config.DELTA) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void validate(int run) {
        if (run == 0) {
            System.out.println(" -- Result Correct? " + validate(numBodies, pos, vel, posRef, velRef));
        } else {
            System.out.println();
        }
    }

    @Override
    int getSize() {
        return numBodies;
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {
        private NBody nbody;
        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            nbody = new NBody(Catalog.DEFAULT.get(Catalog.BenchmarkID.NBody).size());
            executionPlan = nbody.buildExecutionPlan();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void nbodySequential(JMHBenchmark state) {
            state.nbody.computeSequential();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void nbodyParallelStreams(JMHBenchmark state) {
            state.nbody.computeWithJavaStreams();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void nbodyParallelThreads(JMHBenchmark state) throws InterruptedException {
            state.nbody.computeWithJavaThreads();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void nbodyParallelVectorAPI(JMHBenchmark state) {
            state.nbody.computeWithParallelVectorAPI();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void nbodyTornadoVM(JMHBenchmark state) {
            state.executionPlan.execute();
        }
    }

    @Override
    void runWithJMH() throws RunnerException {
        org.openjdk.jmh.runner.options.Options opt = new OptionsBuilder() //
                .include(NBody.class.getName() + ".*") //
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
        return "nbody";
    }

    @Override
    String printSize() {
        return "" + numBodies;
    }

    public static void main(String[] args) throws InterruptedException {
        NBody benchmark = new NBody(Catalog.DEFAULT.get(Catalog.BenchmarkID.NBody).size());
        benchmark.run(args);
    }
}
