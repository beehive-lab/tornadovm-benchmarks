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
package tornadovm.benchmarks.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.OptionalLong;

public class Utils {

    public static Range[] createRangesForCPU(int size) {

        int maxProcessors = Runtime.getRuntime().availableProcessors();
        int chunk = size / maxProcessors;
        int rest = size % maxProcessors;

        Range[] ranges = new Range[maxProcessors];
        for (int i = 0; i < ranges.length; i++) {
            int min = i * chunk;
            int max = i * chunk + chunk;

            // Adjust load
            if (rest > i) {
                max += i + 1;
                min += i;
            } else if (rest != 0) {
                min += rest;
                max += rest;
            }
            ranges[i] = new Range(min, max);
        }

        if (Config.DEBUG) {
            Arrays.stream(ranges).forEach(r -> {
                System.out.println(r + " -- " + (r.max() - r.min()));
            });
        }
        return ranges;
    }

    /**
     * Returns the median of valid timings (nanoseconds), excluding {@code -1} error markers.
     * Returns {@code -1.0} if every entry is an error marker.
     */
    public static double computeMedian(ArrayList<Long> timings) {
        long[] valid = timings.stream().mapToLong(Long::longValue).filter(t -> t != -1L).sorted().toArray();
        if (valid.length == 0) return -1.0;
        int n = valid.length;
        return (n % 2 == 0) ? (valid[n / 2 - 1] + valid[n / 2]) / 2.0 : valid[n / 2];
    }

    /**
     * Returns the minimum of valid timings (nanoseconds), excluding {@code -1} error markers.
     * Returns {@code -1L} if every entry is an error marker.
     */
    public static long computeMin(ArrayList<Long> timings) {
        OptionalLong min = timings.stream().mapToLong(Long::longValue).filter(t -> t != -1L).min();
        return min.isPresent() ? min.getAsLong() : -1L;
    }

    public static void dumpPerformanceTable(ArrayList<ArrayList<Long>> timers, int implementationsToCompare, String benchmarkName, String header) {

        // Create results directory
        File resultDirectory = new File("./results/");
        if (!resultDirectory.exists()) {
            resultDirectory.mkdirs();
        }

        // Print CSV table with RAW elapsed timers
        try (FileWriter fileWriter = new FileWriter("results/" + benchmarkName + "-performanceTable.csv")) {
            // Write header
            fileWriter.write(header);
            // Write data
            for (int i = 0; i < Config.RUNS; i++) {
                StringBuilder builder = new StringBuilder();
                for (int j = 0; j < implementationsToCompare; j++) {
                    if (timers.get(j).size() > i) {
                        builder.append(timers.get(j).get(i)).append(",");
                    } else {
                        // It can be missing due to the selection of a specific implementation
                        builder.append("-1,");
                    }
                }
                fileWriter.write(builder.substring(0, builder.length() - 1));
                fileWriter.write("\n");
            }
        } catch (IOException e) {
            System.err.println("An error occurred: " + e.getMessage());
        }
    }
}
