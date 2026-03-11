/*
 * Copyright (c) 2025-2026, APT Group, Department of Computer Science,
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

import tornadovm.benchmarks.utils.Config;
import tornadovm.benchmarks.utils.Option;
import tornadovm.benchmarks.utils.Utils;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public abstract class BenchmarkDriver extends Benchmark {

    public abstract void computeSequential();

    public abstract void computeWithJavaStreams();

    public abstract void computeWithJavaThreads() throws InterruptedException;

    /**
     * Override this method to reuse the shared fixed-thread-pool across iterations,
     * eliminating per-iteration thread-creation overhead from measurements.
     *
     * <p>The default implementation falls back to {@link #computeWithJavaThreads()},
     * which preserves backward compatibility for subclasses that have not yet been
     * updated. Subclasses should migrate to this form:
     *
     * <pre>{@code
     * @Override
     * protected void computeWithJavaThreadsReusing(ExecutorService executor)
     *         throws InterruptedException {
     *     Range[] ranges = Utils.createRangesForCPU(output.getSize());
     *     List<Future<?>> futures = new ArrayList<>(ranges.length);
     *     for (int t = 0; t < ranges.length; t++) {
     *         final int idx = t;
     *         futures.add(executor.submit(() -> {
     *             for (int j = ranges[idx].min(); j < ranges[idx].max(); j++) {
     *                 // ... work ...
     *             }
     *         }));
     *     }
     *     for (Future<?> f : futures) {
     *         try { f.get(); } catch (ExecutionException e) { throw new RuntimeException(e); }
     *     }
     * }
     * }</pre>
     *
     * @param executor a fixed-size thread pool with one thread per available processor,
     *                 created once before warm-up and shut down after measurement.
     */
    protected void computeWithJavaThreadsReusing(ExecutorService executor) throws InterruptedException {
        computeWithJavaThreads();
    }

    public abstract void computeWithParallelVectorAPI();

    public abstract TornadoExecutionPlan buildExecutionPlan();

    /**
     * Reset output buffers (and, if inputs are mutated in-place, restore them from a
     * pristine copy) so that every measured iteration starts from an identical state.
     * This method is called <em>outside</em> the timed region, before each warm-up and
     * measured iteration.
     */
    public abstract void resetOutputs();

    public abstract void validate(int runID);

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Print a one-line summary for a backend after all its measured iterations finish.
     * Timings are in nanoseconds; the summary converts to milliseconds.
     */
    private void printBackendSummary(String backendName, ArrayList<Long> timings) {
        long errorCount = timings.stream().filter(t -> t == -1L).count();
        int validCount = timings.size() - (int) errorCount;
        if (validCount == 0) {
            System.out.printf("  %-32s  FAILED — all %d iterations errored%n",
                    "[" + backendName + "]", timings.size());
        } else {
            double medianMs = Utils.computeMedian(timings) * 1E-6;
            double minMs    = Utils.computeMin(timings)    * 1E-6;
            System.out.printf("  %-32s  median: %9.3f ms   min: %9.3f ms   (%d/%d valid)%n",
                    "[" + backendName + "]", medianMs, minMs, validCount, timings.size());
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Main benchmark orchestration
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Run all applicable backends, measure their steady-state performance, and dump
     * results to a CSV file.
     *
     * <h3>Measurement protocol</h3>
     * <ol>
     *   <li><b>Warm-up phase</b> ({@code Config.WARMUP_RUNS} iterations, untimed): allows
     *       the JIT compiler, ForkJoinPool, thread-pool, and GPU runtime to reach
     *       steady state before any timing begins.</li>
     *   <li><b>Reset before each iteration</b>: {@code resetOutputs()} is called outside
     *       the timed region so that every measured iteration starts from an identical
     *       input/output state. This is critical for benchmarks that mutate inputs
     *       in-place (e.g. Silu, NBody).</li>
     *   <li><b>Thread pool reuse</b>: a single {@code ExecutorService} is created before
     *       the threads warm-up and reused across all warm-up and measured iterations,
     *       eliminating per-call thread-creation overhead.</li>
     *   <li><b>TornadoVM warm-up</b>: the first {@code executionPlan.execute()} triggers
     *       GPU kernel compilation and initial buffer allocation. These are absorbed in the
     *       warm-up phase; only steady-state timings are reported.</li>
     *   <li><b>Summary stats</b>: median and min are computed after each backend (ignoring
     *       {@code -1} error markers). Raw per-iteration nanosecond timings are also written
     *       to the CSV for offline analysis.</li>
     * </ol>
     *
     * <p>Verbose per-iteration logging can be enabled with {@code Config.VERBOSE = true}.
     */
    @Override
    public void runTestAll(int size, Option option) throws InterruptedException {
        ArrayList<ArrayList<Long>> timers = new ArrayList<>();
        StringBuilder headerTable = new StringBuilder();

        System.out.println(Config.Colours.BLUE
                + "\n=== Benchmark: " + getName() + "  size=" + printSize() + " ==="
                + Config.Colours.RESET);
        System.out.println("  Warm-up iterations  : " + Config.WARMUP_RUNS
                + "  (not included in table)");
        System.out.println("  Measured iterations : " + Config.RUNS);
        System.out.println("  Verbose per-iter log: " + Config.VERBOSE);
        System.out.println("  Reporting           : median (primary), min (secondary)");
        System.out.println("  TornadoVM timings   : steady-state (post-warmup)");
        System.out.println();

        // ── 1. Sequential ────────────────────────────────────────────────────────
        timers.add(new ArrayList<>());
        headerTable.append("sequential");

        if (option == Option.TORNADO_ONLY) {
            // Run once for the reference result used in validation; no warmup needed.
            resetOutputs();
            long start = System.nanoTime();
            computeSequential();
            long end = System.nanoTime();
            timers.getLast().add(end - start);
            // No summary line — this is purely a reference run, not a performance target.
        } else {
            System.out.println(Config.Colours.CYAN
                    + "[Sequential] Warming up (" + Config.WARMUP_RUNS + " iters)..."
                    + Config.Colours.RESET);
            for (int w = 0; w < Config.WARMUP_RUNS; w++) {
                resetOutputs();
                computeSequential();
            }

            System.out.println(Config.Colours.CYAN
                    + "[Sequential] Measuring (" + Config.RUNS + " iters)..."
                    + Config.Colours.RESET);
            for (int i = 0; i < Config.RUNS; i++) {
                resetOutputs();
                long start = System.nanoTime();
                computeSequential();
                long end = System.nanoTime();
                timers.getLast().add(end - start);
                if (Config.VERBOSE) {
                    System.out.printf("  [sequential] iter %3d: %,.3f ms%n", i, (end - start) * 1E-6);
                }
            }
            printBackendSummary("Sequential", timers.getLast());
        }

        if (option == Option.ALL || option == Option.JAVA_ONLY) {

            // ── 2. Parallel Streams ──────────────────────────────────────────────
            timers.add(new ArrayList<>());
            headerTable.append(",streams");

            System.out.println(Config.Colours.CYAN
                    + "[Streams] Warming up (" + Config.WARMUP_RUNS + " iters)..."
                    + Config.Colours.RESET);
            for (int w = 0; w < Config.WARMUP_RUNS; w++) {
                resetOutputs();
                computeWithJavaStreams();
            }

            System.out.println(Config.Colours.CYAN
                    + "[Streams] Measuring (" + Config.RUNS + " iters)..."
                    + Config.Colours.RESET);
            for (int i = 0; i < Config.RUNS; i++) {
                resetOutputs();
                long start = System.nanoTime();
                computeWithJavaStreams();
                long end = System.nanoTime();
                timers.getLast().add(end - start);
                if (Config.VERBOSE) {
                    System.out.printf("  [streams] iter %3d: %,.3f ms%n", i, (end - start) * 1E-6);
                }
                if (i == 0) validate(i);
            }
            printBackendSummary("Streams", timers.getLast());

            // ── 3. Java Threads (shared, reusable thread pool) ──────────────────
            //
            // The executor is created once here and passed to computeWithJavaThreadsReusing().
            // Subclasses that override computeWithJavaThreadsReusing() use executor.submit()
            // instead of creating new threads, removing thread-creation overhead from
            // the timed region. The default implementation falls back to computeWithJavaThreads()
            // for subclasses that have not yet been updated.
            timers.add(new ArrayList<>());
            headerTable.append(",threads");

            int nThreads = Runtime.getRuntime().availableProcessors();
            ExecutorService executor = Executors.newFixedThreadPool(nThreads);
            try {
                System.out.println(Config.Colours.CYAN
                        + "[Threads] Warming up (" + Config.WARMUP_RUNS + " iters, pool size=" + nThreads + ")..."
                        + Config.Colours.RESET);
                for (int w = 0; w < Config.WARMUP_RUNS; w++) {
                    resetOutputs();
                    computeWithJavaThreadsReusing(executor);
                }

                System.out.println(Config.Colours.CYAN
                        + "[Threads] Measuring (" + Config.RUNS + " iters)..."
                        + Config.Colours.RESET);
                for (int i = 0; i < Config.RUNS; i++) {
                    resetOutputs();
                    long start = System.nanoTime();
                    computeWithJavaThreadsReusing(executor);
                    long end = System.nanoTime();
                    timers.getLast().add(end - start);
                    if (Config.VERBOSE) {
                        System.out.printf("  [threads] iter %3d: %,.3f ms%n", i, (end - start) * 1E-6);
                    }
                    if (i == 0) validate(i);
                }
            } finally {
                executor.shutdown();
            }
            printBackendSummary("Threads", timers.getLast());

            // ── 4. Parallel Vector API ───────────────────────────────────────────
            timers.add(new ArrayList<>());
            headerTable.append(",parallelVectorAPI");

            boolean vectorApiFailed = false;
            System.out.println(Config.Colours.CYAN
                    + "[VectorAPI] Warming up (" + Config.WARMUP_RUNS + " iters)..."
                    + Config.Colours.RESET);
            try {
                for (int w = 0; w < Config.WARMUP_RUNS; w++) {
                    resetOutputs();
                    computeWithParallelVectorAPI();
                }
            } catch (RuntimeException e) {
                System.out.println("  [VectorAPI] Warm-up failed: " + e.getMessage()
                        + " — skipping measurement.");
                vectorApiFailed = true;
            }

            System.out.println(Config.Colours.CYAN
                    + "[VectorAPI] Measuring (" + Config.RUNS + " iters)..."
                    + Config.Colours.RESET);
            for (int i = 0; i < Config.RUNS; i++) {
                if (vectorApiFailed) {
                    timers.getLast().add(-1L);
                    continue;
                }
                try {
                    resetOutputs();
                    long start = System.nanoTime();
                    computeWithParallelVectorAPI();
                    long end = System.nanoTime();
                    timers.getLast().add(end - start);
                    if (Config.VERBOSE) {
                        System.out.printf("  [vectorapi] iter %3d: %,.3f ms%n", i, (end - start) * 1E-6);
                    }
                    if (i == 0) validate(i);
                } catch (RuntimeException e) {
                    System.out.println("  [VectorAPI] Error (iter " + i + "): " + e.getMessage());
                    timers.getLast().add(-1L);
                }
            }
            printBackendSummary("Parallel VectorAPI", timers.getLast());
        }

        // ── 5. TornadoVM (GPU) ───────────────────────────────────────────────────
        //
        // Timing reported here is STEADY-STATE: the warm-up phase absorbs the first-time
        // GPU kernel compilation (OpenCL/CUDA JIT) and initial memory buffer allocation.
        // The CSV column "TornadoVM" therefore reflects kernel dispatch + data transfer
        // overhead for a pre-compiled, pre-allocated plan.
        if (option == Option.ALL || option == Option.TORNADO_ONLY) {
            try (TornadoExecutionPlan executionPlan = buildExecutionPlan()) {
                String deviceProp = System.getProperty("benchmark.device", "0:0");
                String[] parts = deviceProp.split(":");
                TornadoDevice device = TornadoExecutionPlan.getDevice(
                        Integer.parseInt(parts[0]), Integer.parseInt(parts[1]));
                executionPlan.withDevice(device);

                timers.add(new ArrayList<>());
                headerTable.append(",TornadoVM");

                System.out.println(Config.Colours.CYAN
                        + "[TornadoVM] Warming up (" + Config.WARMUP_RUNS
                        + " iters) — iter 0 triggers GPU kernel compilation..."
                        + Config.Colours.RESET);
                for (int w = 0; w < Config.WARMUP_RUNS; w++) {
                    resetOutputs();
                    executionPlan.execute();
                }

                System.out.println(Config.Colours.CYAN
                        + "[TornadoVM] Measuring steady-state (" + Config.RUNS + " iters)..."
                        + Config.Colours.RESET);
                for (int i = 0; i < Config.RUNS; i++) {
                    resetOutputs();
                    long start = System.nanoTime();
                    executionPlan.execute();
                    long end = System.nanoTime();
                    timers.getLast().add(end - start);
                    if (Config.VERBOSE) {
                        System.out.printf("  [tornadovm] iter %3d: %,.3f ms%n", i, (end - start) * 1E-6);
                    }
                    if (i == 0) validate(i);
                }
                printBackendSummary("TornadoVM (steady-state)", timers.getLast());

            } catch (TornadoExecutionPlanException e) {
                throw new RuntimeException(e);
            }
        }

        System.out.println();
        Utils.dumpPerformanceTable(timers, timers.size(), getName(), headerTable.append("\n").toString());
    }
}
