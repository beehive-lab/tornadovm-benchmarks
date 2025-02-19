package tornadovm.benchmarks;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.openjdk.jmh.annotations.Benchmark;
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
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DFT implements TornadoBenchmark {

    final static int SIZE = 8192;
    final int RUNS = 10;

    public static void computeSequential(FloatArray inreal, FloatArray inimag, FloatArray outreal, FloatArray outimag) {
        int n = inreal.getSize();
        for (int k = 0; k < n; k++) { // For each output element
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

    public static void computeWithTornado(FloatArray inreal, FloatArray inimag, FloatArray outreal, FloatArray outimag) {
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

    public void computeWithJavaStreams(FloatArray inreal, FloatArray inimag, FloatArray outreal, FloatArray outimag) {
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

    public void computeWithParallelVectorAPI(FloatArray inreal, FloatArray inimag, FloatArray outreal, FloatArray outimag) {
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

    public void computeWithJavaThreads(FloatArray inreal, FloatArray inimag, FloatArray outreal, FloatArray outimag) throws InterruptedException {
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

    public boolean validate(int size, FloatArray outRealSeq, FloatArray outImagSeq, FloatArray outReal, FloatArray outImag) {
        boolean val = true;
        for (int i = 0; i < size; i++) {
            if (Math.abs(outRealSeq.get(i) - outReal.get(i)) > 1.0f) {
                System.out.println(outReal.get(i) + " vs " + outRealSeq.get(i) + "\n");
                val = false;
                break;
            }
            if (Math.abs(outImagSeq.get(i) - outImag.get(i)) > 1.0f) {
                System.out.println(outImagSeq.get(i) + " vs " + outImag.get(i) + "\n");
                val = false;
                break;
            }
        }
        return val;
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {

        private DFT dft;

        private FloatArray inReal;
        private FloatArray inImag;
        private FloatArray outReal;
        private FloatArray outImag;

        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            final int size = SIZE;
            dft = new DFT();

            inReal = new FloatArray(size);
            inImag = new FloatArray(size);
            outReal = new FloatArray(size);
            outImag = new FloatArray(size);

            Random r = new Random();
            for (int i = 0; i < size; i++) {
                inReal.set(i, 1 / r.nextFloat());
                inImag.set(i, 1 / r.nextFloat());
            }

            FloatArray outRealTornado = new FloatArray(size);
            FloatArray outImagTornado = new FloatArray(size);

            TaskGraph taskGraph = new TaskGraph("benchmark")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, inReal, inImag)
                    .task("dft", DFT::computeWithTornado, inReal, inImag, outRealTornado, outImagTornado)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outRealTornado, outImagTornado);
            executionPlan = new TornadoExecutionPlan(taskGraph.snapshot());
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void dftSequential(JMHBenchmark state) {
            computeSequential(state.inReal, state.inImag, state.outReal, state.outImag);
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void dftParallelStreams(JMHBenchmark state) {
            state.dft.computeWithJavaStreams(state.inReal, state.inImag, state.outReal, state.outImag);
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void dftParallelThreads(JMHBenchmark state) {
            try {
                state.dft.computeWithJavaThreads(state.inReal, state.inImag, state.outReal, state.outImag);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void dftParallelVectorAPI(JMHBenchmark state) {
            state.dft.computeWithParallelVectorAPI(state.inReal, state.inImag, state.outReal, state.outImag);
        }

        @Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void dftTornadoVM(MatrixMultiplication.JMHBenchmark state) {
            state.executionPlan.execute();
        }
    }

    private static void runWithJMH() throws RunnerException {
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

    private void runTestAll(final int size, Option option) throws InterruptedException {

        FloatArray inReal = new FloatArray(size);
        FloatArray inImag = new FloatArray(size);
        FloatArray outRealSeq = new FloatArray(size);
        FloatArray outImagSeq = new FloatArray(size);

        Random r = new Random(71);
        for (int i = 0; i < size; i++) {
            inReal.set(i, 1 / r.nextFloat());
            inImag.set(i, 1 / r.nextFloat());
        }

        // 5 implementations to compare
        final int implementationsToCompare = 5;
        ArrayList<ArrayList<Long>> timers = IntStream.range(0, implementationsToCompare) //
                .<ArrayList<Long>>mapToObj(i -> new ArrayList<>()) //
                .collect(Collectors.toCollection(ArrayList::new));

        for (int i = 0; i < RUNS; i++) {
            long start = System.nanoTime();
            computeSequential(inReal, inImag, outRealSeq, outImagSeq);
            long end = System.nanoTime();
            long elapsedTime = (end - start);
            timers.get(0).add(elapsedTime);
            double elapsedTimeMilliseconds = elapsedTime * 1E-6;

            System.out.println("Elapsed time: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");

            if (option == Option.TORNADO_ONLY) {
                // We only run one iteration just to run the reference implementation to check results.
                break;
            }
        }

        if (option == Option.ALL || option == Option.JAVA_ONLY) {
            // 2. Parallel Streams
            FloatArray outRealStream = new FloatArray(size);
            FloatArray outImagStream = new FloatArray(size);
            for (int i = 0; i < RUNS; i++) {
                long start = System.nanoTime();
                computeWithJavaStreams(inReal, inImag, outRealStream, outImagStream);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(1).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Stream Elapsed time: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                System.out.println(" -- Result Correct? " + validate(SIZE, outRealSeq, outImagSeq, outRealStream, outImagStream));
            }

            // 3. Parallel with Java Threads
            FloatArray outRealThreads = new FloatArray(size);
            FloatArray outImagThreads = new FloatArray(size);
            for (int i = 0; i < RUNS; i++) {
                long start = System.nanoTime();
                computeWithJavaThreads(inReal, inImag, outRealThreads, outImagThreads);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(2).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Elapsed time Threads: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                System.out.println(" -- Result Correct? " + validate(SIZE, outRealSeq, outImagSeq, outRealThreads, outImagThreads));
            }

            // 4. Parallel with Java Vector API
            FloatArray outRealVector = new FloatArray(size);
            FloatArray outImagVector = new FloatArray(size);
            for (int i = 0; i < RUNS; i++) {
                long start = System.nanoTime();
                computeWithParallelVectorAPI(inReal, inImag, outRealVector, outImagVector);
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.get(3).add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Elapsed time Parallel Vectorized: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                System.out.println(" -- Result Correct? " + validate(SIZE, outRealSeq, outImagSeq, outRealVector, outImagVector));
            }
        }


        if (option == Option.ALL || option == Option.TORNADO_ONLY) {
            // TornadoVM
            FloatArray outRealTornado = new FloatArray(size);
            FloatArray outImagTornado = new FloatArray(size);

            TaskGraph taskGraph = new TaskGraph("benchmark")
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION, inReal, inImag)
                    .task("dft", DFT::computeWithTornado, inReal, inImag, outRealTornado, outImagTornado)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, outRealTornado, outImagTornado);
            try (TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(taskGraph.snapshot())) {

                // 5. On the GPU using TornadoVM
                for (int i = 0; i < RUNS; i++) {
                    long start = System.nanoTime();
                    executionPlan.execute();
                    long end = System.nanoTime();
                    long elapsedTime = (end - start);
                    timers.get(4).add(elapsedTime);
                    double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                    System.out.print("Elapsed time TornadoVM-GPU: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                    System.out.println(" -- Result Correct? " + validate(SIZE, outRealSeq, outImagSeq, outRealTornado, outImagTornado));
                }
            } catch (TornadoExecutionPlanException e) {
                throw new RuntimeException(e);
            }
        }

        if (option == Option.ALL) {
            // Print CSV table with RAW elapsed timers
            try (FileWriter fileWriter = new FileWriter("dft-performanceTable.csv")) {
                // Write header
                fileWriter.write("sequential,streams,threads,TornadoVM\n");
                // Write data
                for (int i = 0; i < RUNS; i++) {
                    StringBuilder builder = new StringBuilder();
                    for (int j = 0; j < implementationsToCompare; j++) {
                        builder.append(timers.get(j).get(i)).append(",");
                    }
                    fileWriter.write(builder.substring(0, builder.length() - 1));
                    fileWriter.write("\n");
                }
            } catch (IOException e) {
                System.err.println("An error occurred: " + e.getMessage());
            }
        }

    }

    @Override
    public void run(String[] args) {
        System.out.println("[INFO] DFT");
        final int size = SIZE;
        System.out.println("[INFO] DFT size: " + size);

        Option option = Option.ALL;
        if (args.length > 0) {
            switch (args[0]) {
                case "jmh" -> {
                    try {
                        runWithJMH();
                        return;
                    } catch (Exception e) {
                        System.err.println("An error occurred: " + e.getMessage());
                    }
                }
                case "onlyJavaSeq" -> option = Option.JAVA_SEQ_ONLY;
                case "onlyJava" -> option = Option.JAVA_ONLY;
                case "onlyTornadoVM" -> option = Option.TORNADO_ONLY;
            }
        }
        try {
            runTestAll(size, option);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        DFT benchmark = new DFT();
        benchmark.run(args);
    }
}
