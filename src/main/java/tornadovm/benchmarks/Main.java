package tornadovm.benchmarks;

import java.util.Arrays;

/**
 * How to run?
 *
 * <p>
 * <code>
 *    java -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.Main
 * </code>
 * </p>
 */
public class Main {

    public static void main(String[] args) {

        if (args.length > 0) {
            TornadoBenchmark benchmark;
            String benchmarkName = args[0];

            // remove element 0 from the list
            String[] arguments = new String[args.length - 1];
            System.arraycopy(args, 1, arguments, 0, arguments.length);
            switch (benchmarkName) {
                case "mxm" -> benchmark = new MatrixMultiplication();
                case "dft" -> benchmark = new DFT();
                case "montecarlo" -> benchmark = new Montecarlo();
                default -> throw new IllegalArgumentException("Invalid benchmark: " + benchmarkName);
            }
            benchmark.run(arguments);
        } else {
            System.out.println("[TornadoVM Benchmarks] Running all benchmarks...");

            TornadoBenchmark[] benchmarks = new TornadoBenchmark[3];
            benchmarks[0] = new MatrixMultiplication();
            benchmarks[1] = new DFT();
            benchmarks[2] = new Montecarlo();

            Arrays.stream(benchmarks).forEach(benchmark -> benchmark.run(args));
        }
    }
}
