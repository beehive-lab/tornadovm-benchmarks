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
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class BlurFilter extends BenchmarkDriver {

    public static final int FILTER_WIDTH = 31;

    private BufferedImage image;

    int w;
    int h;
    IntArray redChannel;
    IntArray greenChannel;
    IntArray blueChannel;
    IntArray alphaChannel;
    IntArray redFilter;
    IntArray greenFilter;
    IntArray blueFilter;

    IntArray redFilterRef;
    IntArray greenFilterRef;
    IntArray blueFilterRef;

    FloatArray filter;

    public BlurFilter(BufferedImage image) {
        this.image = image;
        initData();
    }

    private void initData() {
        w = image.getWidth();
        h = image.getHeight();

        redChannel = new IntArray(w * h);
        greenChannel = new IntArray(w * h);
        blueChannel = new IntArray(w * h);
        alphaChannel = new IntArray(w * h);

        redFilter = new IntArray(w * h);
        greenFilter = new IntArray(w * h);
        blueFilter = new IntArray(w * h);

        redFilterRef= new IntArray(w * h);
        greenFilterRef = new IntArray(w * h);
        blueFilterRef = new IntArray(w * h);

        filter = new FloatArray(w * h);
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                filter.set(i * h + j,  1.f / (FILTER_WIDTH * FILTER_WIDTH));
            }
        }
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                int rgb = image.getRGB(i, j);
                alphaChannel.set(i * h + j,  (rgb >> 24) & 0xFF);
                redChannel.set(i * h + j,  (rgb >> 16) & 0xFF);
                greenChannel.set(i * h + j,  (rgb >> 8) & 0xFF);
                blueChannel.set(i * h + j,  (rgb & 0xFF));
            }
        }
    }

    private static void channelConvolutionSequential(IntArray channel, IntArray channelBlurred, final int numRows, final int numCols, FloatArray filter, final int filterWidth) {
        for (int r = 0; r < numRows; r++) {
            for (int c = 0; c < numCols; c++) {
                float result = 0.0f;
                for (int filter_r = -filterWidth / 2; filter_r <= filterWidth / 2; filter_r++) {
                    for (int filter_c = -filterWidth / 2; filter_c <= filterWidth / 2; filter_c++) {
                        int image_r = Math.min(Math.max(r + filter_r, 0), (numRows - 1));
                        int image_c = Math.min(Math.max(c + filter_c, 0), (numCols - 1));
                        float image_value = channel.get(image_r * numCols + image_c);
                        float filter_value = filter.get((filter_r + filterWidth / 2) * filterWidth + filter_c + filterWidth / 2);
                        result += image_value * filter_value;
                    }
                }
                int finalValue = result > 255 ? 255 : (int) result;
                channelBlurred.set(r * numCols + c, finalValue);
            }
        }
    }

    private static void channelConvolutionStreams(IntArray channel, IntArray channelBlurred, final int numRows, final int numCols, FloatArray filter, final int filterWidth) {
        IntStream.range(0, numRows).parallel().forEach(r ->  {
            IntStream.range(0, numCols).parallel().forEach(c -> {
                float result = 0.0f;
                for (int filter_r = -filterWidth / 2; filter_r <= filterWidth / 2; filter_r++) {
                    for (int filter_c = -filterWidth / 2; filter_c <= filterWidth / 2; filter_c++) {
                        int image_r = Math.min(Math.max(r + filter_r, 0), (numRows - 1));
                        int image_c = Math.min(Math.max(c + filter_c, 0), (numCols - 1));
                        float image_value = channel.get(image_r * numCols + image_c);
                        float filter_value = filter.get((filter_r + filterWidth / 2) * filterWidth + filter_c + filterWidth / 2);
                        result += image_value * filter_value;
                    }
                }
                int finalValue = result > 255 ? 255 : (int) result;
                channelBlurred.set(r * numCols + c, finalValue);
            });
        });
    }

    private void channelConvolutionThreads(IntArray channel, IntArray channelBlurred, final int numRows, final int numCols, FloatArray filter, final int filterWidth) throws InterruptedException {
        Range[] ranges = Utils.createRangesForCPU(w);
        int maxProcessors = Runtime.getRuntime().availableProcessors();
        Thread[] threads = new Thread[maxProcessors];
        IntStream.range(0, threads.length).forEach(threadIndex -> {
            threads[threadIndex] = new Thread(() -> {
                for (int i = ranges[threadIndex].min(); i < ranges[threadIndex].max(); i++) {
                    for (int c = 0; c < numCols; c++) {
                        float result = 0.0f;
                        for (int filter_r = -filterWidth / 2; filter_r <= filterWidth / 2; filter_r++) {
                            for (int filter_c = -filterWidth / 2; filter_c <= filterWidth / 2; filter_c++) {
                                int image_r = Math.min(Math.max(i + filter_r, 0), (numRows - 1));
                                int image_c = Math.min(Math.max(c + filter_c, 0), (numCols - 1));
                                float image_value = channel.get(image_r * numCols + image_c);
                                float filter_value = filter.get((filter_r + filterWidth / 2) * filterWidth + filter_c + filterWidth / 2);
                                result += image_value * filter_value;
                            }
                        }
                        int finalValue = result > 255 ? 255 : (int) result;
                        channelBlurred.set(i * numCols + c, finalValue);
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
    public void computeSequential() {
        channelConvolutionSequential(redChannel, redFilterRef, w, h, filter, FILTER_WIDTH);
        channelConvolutionSequential(greenChannel, greenFilterRef, w, h, filter, FILTER_WIDTH);
        channelConvolutionSequential(blueChannel, blueFilterRef, w, h, filter, FILTER_WIDTH);
    }

    @Override
    public void computeWithJavaStreams() {
        channelConvolutionStreams(redChannel, redFilter, w, h, filter, FILTER_WIDTH);
        channelConvolutionStreams(greenChannel, greenFilter, w, h, filter, FILTER_WIDTH);
        channelConvolutionStreams(blueChannel, blueFilter, w, h, filter, FILTER_WIDTH);
    }

    @Override
    public void computeWithJavaThreads() throws InterruptedException {
        channelConvolutionThreads(redChannel, redFilter, w, h, filter, FILTER_WIDTH);
        channelConvolutionThreads(greenChannel, greenFilter, w, h, filter, FILTER_WIDTH);
        channelConvolutionThreads(blueChannel, blueFilter, w, h, filter, FILTER_WIDTH);
    }

    @Override
    public void computeWithParallelVectorAPI() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    private static void compute(IntArray channel, IntArray channelBlurred, final int numRows, final int numCols, FloatArray filter, final int filterWidth) {
        for (@Parallel int r = 0; r < numRows; r++) {
            for (@Parallel int c = 0; c < numCols; c++) {
                float result = 0.0f;
                for (int filter_r = -filterWidth / 2; filter_r <= filterWidth / 2; filter_r++) {
                    for (int filter_c = -filterWidth / 2; filter_c <= filterWidth / 2; filter_c++) {
                        int image_r = Math.min(Math.max(r + filter_r, 0), (numRows - 1));
                        int image_c = Math.min(Math.max(c + filter_c, 0), (numCols - 1));
                        float image_value = channel.get(image_r * numCols + image_c);
                        float filter_value = filter.get((filter_r + filterWidth / 2) * filterWidth + filter_c + filterWidth / 2);
                        result += image_value * filter_value;
                    }
                }
                int finalValue = result > 255 ? 255 : (int) result;
                channelBlurred.set(r * numCols + c, finalValue);
            }
        }
    }

    @Override
    public TornadoExecutionPlan buildExecutionPlan() {
        TaskGraph taskGraph = new TaskGraph("blur") //
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, redChannel, greenChannel, blueChannel, filter) //
                .task("red", BlurFilter::compute, redChannel, redFilter, w, h, filter, FILTER_WIDTH) //
                .task("green", BlurFilter::compute, greenChannel, greenFilter, w, h, filter, FILTER_WIDTH) //
                .task("blue", BlurFilter::compute, blueChannel, blueFilter, w, h, filter, FILTER_WIDTH) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, redFilter, greenFilter, blueFilter);
        return new TornadoExecutionPlan(taskGraph.snapshot());
    }

    @Override
    public void resetOutputs() {
        redFilter.init(0);
        greenFilter.init(0);
        blueFilter.init(0);
    }

    private boolean validate(IntArray redFilterRef, IntArray redFilter, IntArray greenFilterRef, IntArray greenFilter, IntArray blueFilterRef, IntArray blueFilter) {
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                Color colorComputed = new Color(redFilter.get(i * h + j), greenFilter.get(i * h + j), blueFilter.get(i * h + j), alphaChannel.get(i * h + j));
                Color colorRef = new Color(redFilterRef.get(i * h + j), greenFilterRef.get(i * h + j), blueFilterRef.get(i * h + j), alphaChannel.get(i * h + j));
                if (colorRef.getRGB() != colorComputed.getRGB()) {
                    return false;
                }
            }
        }
        return true;
    }

    @Override
    public void validate(int run) {
        if (run == 0) {
            System.out.println(" -- Result Correct? " + validate(redFilterRef, redFilter, greenFilterRef, greenFilter, blueFilterRef, blueFilter));
        } else {
            System.out.println();
        }
    }



    @Override
    int getSize() {
        return w * h;
    }

    @State(Scope.Thread)
    public static class JMHBenchmark {

        private BlurFilter blurFilter;
        private TornadoExecutionPlan executionPlan;

        @Setup(Level.Trial)
        public void doSetup() {
            blurFilter = new BlurFilter(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixTranspose).image());
            executionPlan = blurFilter.buildExecutionPlan();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void blurFilterSequential(JMHBenchmark state) {
            state.blurFilter.computeSequential();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void blurFilterParallelStreams(JMHBenchmark state) {
            state.blurFilter.computeWithJavaStreams();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void blurFilterParallelThreads(JMHBenchmark state) throws InterruptedException {
            state.blurFilter.computeWithJavaThreads();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void blurFilterParallelVectorAPI(JMHBenchmark state) {
            state.blurFilter.computeWithParallelVectorAPI();
        }

        @org.openjdk.jmh.annotations.Benchmark
        @BenchmarkMode(Mode.AverageTime)
        @Warmup(iterations = 2, time = 60)
        @Measurement(iterations = 5, time = 30)
        @OutputTimeUnit(TimeUnit.NANOSECONDS)
        @Fork(1)
        public void blurFilterTornadoVM(JMHBenchmark state) {
            state.executionPlan.execute();
        }
    }

    @Override
    void runWithJMH() throws RunnerException {
        org.openjdk.jmh.runner.options.Options opt = new OptionsBuilder() //
                .include(BlurFilter.class.getName() + ".*") //
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
        return "blurFilter";
    }

    @Override
    String printSize() {
        return w + "x" + h;
    }

    public static void main(String[] args) throws InterruptedException {
        BlurFilter benchmark = new BlurFilter(Catalog.DEFAULT.get(Catalog.BenchmarkID.BlurFilter).image());
        benchmark.run(args);
    }
}
