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

import uk.ac.manchester.tornado.api.TornadoBackend;
import uk.ac.manchester.tornado.api.TornadoDeviceMap;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.TornadoDeviceType;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

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
                    builder.append(timers.get(j).get(i)).append(",");
                }
                fileWriter.write(builder.substring(0, builder.length() - 1));
                fileWriter.write("\n");
            }
        } catch (IOException e) {
            System.err.println("An error occurred: " + e.getMessage());
        }
    }

    public static TornadoDevice getDeviceByType(TornadoDeviceType deviceType) {
        TornadoDeviceMap deviceMap = TornadoExecutionPlan.getTornadoDeviceMap();
        List<TornadoBackend> backendStream = deviceMap
                .getBackendsWithPredicate(backend -> backend
                        .getAllDevices()
                        .stream()
                        .anyMatch(device -> device.getDeviceType() == deviceType));

        if (backendStream.isEmpty()) {
            Stream<TornadoDevice> tornadoDeviceStream = backendStream.getFirst()
                    .getAllDevices()
                    .stream()
                    .filter(device -> device.getDeviceType() == deviceType);
            // Return first device that meets the criteria
            return tornadoDeviceStream.findFirst().get();
        } else {
            return null;
        }
    }

    public static TornadoDevice getDeviceByName(String deviceName) {
        TornadoDeviceMap deviceMap = TornadoExecutionPlan.getTornadoDeviceMap();
        List<TornadoBackend> backendStream = deviceMap
                .getBackendsWithPredicate(backend -> backend
                        .getAllDevices()
                        .stream()
                        .anyMatch(device -> device.getPhysicalDevice().getDeviceName().toLowerCase().contains(deviceName.toLowerCase())));

        if (!backendStream.isEmpty()) {
            Stream<TornadoDevice> tornadoDeviceStream = backendStream.getFirst()
                    .getAllDevices()
                    .stream()
                    .filter(device -> device.getPhysicalDevice().getDeviceName().toLowerCase().contains(deviceName.toLowerCase()));
            // Return first device that meets the criteria
            return tornadoDeviceStream.findFirst().get();
        }
        return null;
    }
}
