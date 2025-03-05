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

    public static void main(String[] args) throws InterruptedException {

        if (args.length > 0) {
            Benchmark benchmark;
            String benchmarkName = args[0];

            switch (benchmarkName) {
                case "mxm" -> benchmark = new MatrixMultiplication(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixMul).size());
                case "dft" -> benchmark = new DFT(Catalog.DEFAULT.get(Catalog.BenchmarkID.DFT).size());
                case "montecarlo" -> benchmark = new Montecarlo(Catalog.DEFAULT.get(Catalog.BenchmarkID.Montecarlo).size());
                case "mandelbrot" -> benchmark = new Mandelbrot(Catalog.DEFAULT.get(Catalog.BenchmarkID.Mandelbrot).size());
                case "mxv" -> benchmark = new MatrixVector(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixVector).size());
                case "mt" -> benchmark = new MatrixTranspose(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixTranspose).size());
                case "blackscholes" -> benchmark = new Blackscholes(Catalog.DEFAULT.get(Catalog.BenchmarkID.Blackscholes).size());
                case "blurfilter" -> benchmark = new BlurFilter(Catalog.DEFAULT.get(Catalog.BenchmarkID.BlurFilter).image());
                case "saxpy" -> benchmark = new Saxpy(Catalog.DEFAULT.get(Catalog.BenchmarkID.Saxpy).size());
                default -> throw new IllegalArgumentException("Invalid benchmark: " + benchmarkName);
            }
            // remove element 0 from the list
            String[] arguments = new String[args.length - 1];
            System.arraycopy(args, 1, arguments, 0, arguments.length);
            benchmark.run(arguments);
        } else {
            System.out.println("[TornadoVM Benchmarks] Running all benchmarks...");

            Benchmark[] benchmarks = new Benchmark[Catalog.BenchmarkID.values().length];
            benchmarks[0] = new MatrixMultiplication(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixMul).size());
            benchmarks[1] = new DFT(Catalog.DEFAULT.get(Catalog.BenchmarkID.DFT).size());
            benchmarks[2] = new Montecarlo(Catalog.DEFAULT.get(Catalog.BenchmarkID.Montecarlo).size());
            benchmarks[3] = new Mandelbrot(Catalog.DEFAULT.get(Catalog.BenchmarkID.Mandelbrot).size());
            benchmarks[4] = new MatrixVector(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixVector).size());
            benchmarks[5] = new MatrixTranspose(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixTranspose).size());
            benchmarks[6] = new Blackscholes(Catalog.DEFAULT.get(Catalog.BenchmarkID.Blackscholes).size());
            benchmarks[6] = new BlurFilter(Catalog.DEFAULT.get(Catalog.BenchmarkID.BlurFilter).image());
            benchmarks[7] = new Saxpy(Catalog.DEFAULT.get(Catalog.BenchmarkID.Saxpy).size());

            Arrays.stream(benchmarks).forEach(benchmark -> {
                try {
                    benchmark.run(args);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            });
        }
    }
}
