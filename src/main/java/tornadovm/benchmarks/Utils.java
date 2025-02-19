package tornadovm.benchmarks;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

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

    public static void dumpPerformanceTable(ArrayList<ArrayList<Long>> timers, int implementationsToCompare, String benchmarkName) {
        // Print CSV table with RAW elapsed timers
        try (FileWriter fileWriter = new FileWriter(benchmarkName + "-performanceTable.csv")) {
            // Write header
            fileWriter.write("sequential,streams,threads,vectorAPI,TornadoVM\n");
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
}
