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
package tornadovm.benchmarks.benchmarks;

import tornadovm.benchmarks.utils.Config;
import tornadovm.benchmarks.utils.Option;
import tornadovm.benchmarks.utils.Utils;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;

import java.util.ArrayList;

public abstract class BenchmarkDriver extends Benchmark {

    public abstract void computeSequential();

    public abstract void computeWithJavaStreams();

    public abstract void computeWithJavaThreads() throws InterruptedException;

    public abstract void computeWithParallelVectorAPI();

    public abstract TornadoExecutionPlan buildExecutionPlan();

    public abstract void resetOutputs();

    public abstract void validate(int runID);

    @Override
    public void runTestAll(int size, Option option) throws InterruptedException {
        ArrayList<ArrayList<Long>> timers = new ArrayList<>();
        StringBuilder headerTable = new StringBuilder();

        // 1. Sequential
        timers.add(new ArrayList<>());
        headerTable.append("sequential");
        for (int i = 0; i < Config.RUNS; i++) {
            long start = System.nanoTime();
            computeSequential();
            long end = System.nanoTime();
            long elapsedTime = (end - start);
            timers.getLast().add(elapsedTime);
            double elapsedTimeMilliseconds = elapsedTime * 1E-6;

            System.out.println("Elapsed time: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) ");

            if (option == Option.TORNADO_ONLY) {
                // We only run one iteration just to run the reference implementation to check results.
                break;
            }
        }

        resetOutputs();
        if (option == Option.ALL || option == Option.JAVA_ONLY) {
            // 2. Parallel Streams
            timers.add(new ArrayList<>());
            headerTable.append(",streams");
            for (int i = 0; i < Config.RUNS; i++) {
                long start = System.nanoTime();
                computeWithJavaStreams();
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.getLast().add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Stream Elapsed time: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                validate(i);
            }

            // 3. Parallel with Java Threads
            resetOutputs();
            timers.add(new ArrayList<>());
            headerTable.append(",threads");
            for (int i = 0; i < Config.RUNS; i++) {
                long start = System.nanoTime();
                computeWithJavaThreads();
                long end = System.nanoTime();
                long elapsedTime = (end - start);
                timers.getLast().add(elapsedTime);
                double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                System.out.print("Elapsed time Threads: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                validate(i);
            }

            // 4. Parallel with Java Vector API
            resetOutputs();
            timers.add(new ArrayList<>());
            headerTable.append(",parallelVectorAPI");
            for (int i = 0; i < Config.RUNS; i++) {
                try {
                    long start = System.nanoTime();
                    computeWithParallelVectorAPI();
                    long end = System.nanoTime();
                    long elapsedTime = (end - start);
                    timers.getLast().add(elapsedTime);
                    double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                    System.out.print("Elapsed time Parallel Vectorized: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                    validate(i);
                } catch (RuntimeException e) {
                    System.out.println("Error - Parallel Vector API: " + e.getMessage());
                    // We store -1 in the timers list to indicate that an error has occurred.
                    timers.getLast().add((long) -1);
                }
            }
        }

        if (option == Option.ALL || option == Option.TORNADO_ONLY) {
            try (TornadoExecutionPlan executionPlan = buildExecutionPlan()) {
                resetOutputs();
                timers.add(new ArrayList<>());
                headerTable.append(",TornadoVM");
                TornadoDevice device = TornadoExecutionPlan.getDevice(0, 0);
                executionPlan.withDevice(device);

                // 5. On the GPU using TornadoVM
                for (int i = 0; i < Config.RUNS; i++) {
                    long start = System.nanoTime();
                    executionPlan.execute();
                    long end = System.nanoTime();
                    long elapsedTime = (end - start);
                    timers.getLast().add(elapsedTime);
                    double elapsedTimeMilliseconds = elapsedTime * 1E-6;

                    System.out.print("Elapsed time TornadoVM-GPU: " + (elapsedTime) + " (ns)  -- " + elapsedTimeMilliseconds + " (ms) -- ");
                    validate(i);
                }
            } catch (TornadoExecutionPlanException e) {
                throw new RuntimeException(e);
            }
        }

        Utils.dumpPerformanceTable(timers, timers.size(), getName(), headerTable.append("\n").toString());
    }

}
