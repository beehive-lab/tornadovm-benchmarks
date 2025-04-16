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

import tornadovm.benchmarks.benchmarks.Benchmark;
import tornadovm.benchmarks.benchmarks.Blackscholes;
import tornadovm.benchmarks.benchmarks.BlurFilter;
import tornadovm.benchmarks.benchmarks.DFT;
import tornadovm.benchmarks.benchmarks.JuliaSets;
import tornadovm.benchmarks.benchmarks.Mandelbrot;
import tornadovm.benchmarks.benchmarks.MatrixMultiplication;
import tornadovm.benchmarks.benchmarks.MatrixMultiplicationFP16;
import tornadovm.benchmarks.benchmarks.MatrixTranspose;
import tornadovm.benchmarks.benchmarks.MatrixVector;
import tornadovm.benchmarks.benchmarks.Montecarlo;
import tornadovm.benchmarks.benchmarks.NBody;
import tornadovm.benchmarks.benchmarks.RMSNorm;
import tornadovm.benchmarks.benchmarks.Saxpy;
import tornadovm.benchmarks.benchmarks.Silu;
import tornadovm.benchmarks.benchmarks.SoftMax;
import tornadovm.benchmarks.utils.Catalog;
import tornadovm.benchmarks.utils.Config;

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

    private static Benchmark instanceBenchmark(String benchmarkName) {
        return switch (benchmarkName) {
            case "mxm" -> new MatrixMultiplication(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixMul).size());
            case "dft" -> new DFT(Catalog.DEFAULT.get(Catalog.BenchmarkID.DFT).size());
            case "montecarlo" -> new Montecarlo(Catalog.DEFAULT.get(Catalog.BenchmarkID.Montecarlo).size());
            case "mandelbrot" -> new Mandelbrot(Catalog.DEFAULT.get(Catalog.BenchmarkID.Mandelbrot).size());
            case "mxv" -> new MatrixVector(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixVector).size());
            case "mt" -> new MatrixTranspose(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixTranspose).size());
            case "blackscholes" -> new Blackscholes(Catalog.DEFAULT.get(Catalog.BenchmarkID.Blackscholes).size());
            case "blurfilter" -> new BlurFilter(Catalog.DEFAULT.get(Catalog.BenchmarkID.BlurFilter).image());
            case "saxpy" -> new Saxpy(Catalog.DEFAULT.get(Catalog.BenchmarkID.Saxpy).size());
            case "nbody" -> new NBody(Catalog.DEFAULT.get(Catalog.BenchmarkID.NBody).size());
            case "juliaset" -> new JuliaSets(Catalog.DEFAULT.get(Catalog.BenchmarkID.JuliaSets).size());
            case "rmsnorm" -> new RMSNorm(Catalog.DEFAULT.get(Catalog.BenchmarkID.RMSNORM).size());
            case "mxmfp16" ->
                    new MatrixMultiplicationFP16(Catalog.DEFAULT.get(Catalog.BenchmarkID.MatrixMulFP16).size());
            case "softmax" -> new SoftMax(Catalog.DEFAULT.get(Catalog.BenchmarkID.SoftMax).size());
            case "silu" -> new Silu(Catalog.DEFAULT.get(Catalog.BenchmarkID.Silu).size());
            default -> throw new IllegalArgumentException("Invalid benchmark: " + benchmarkName);
        };
    }

    public static void main(String[] args) throws InterruptedException {

        if (args.length > 0) {
            String benchmarkName = args[0];
            Benchmark benchmark = instanceBenchmark(benchmarkName);
            // remove element 0 from the list
            String[] arguments = new String[args.length - 1];
            System.arraycopy(args, 1, arguments, 0, arguments.length);
            benchmark.run(arguments);
        } else {
            System.out.println(Config.Colours.GREEN + "[INFO] Running all benchmarks..." + Config.Colours.RESET);

            String[] benchmarkNames = new String[] {
                    "mxm", //
                    "dft", //
                    "montecarlo", //
                    "mandelbrot", //
                    "mxv", //
                    "mt", //
                    "blackscholes", //
                    "blurfilter", //
                    "saxpy",//
                    "nbody",//
                    "juliaset",//
                    "rmsnorm",//
                    "mxmfp16",//
                    "softmax",//
                    "silu"//
            };

            for (String benchmarkName : benchmarkNames) {
                Benchmark benchmarks = instanceBenchmark(benchmarkName);
                benchmarks.run(args);
            }
        }
    }
}
