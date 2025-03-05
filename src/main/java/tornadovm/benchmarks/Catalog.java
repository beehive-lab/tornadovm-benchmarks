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

import java.util.HashMap;
import java.util.Map;

public class Catalog {

    public static final Map<BenchmarkID, DefaultCatalog> DEFAULT = new HashMap<>();

    public record DefaultCatalog(int dimensions, int size) {}

    public enum BenchmarkID {
        Blackscholes("Blackscholes"),
        DFT("dft"),
        Mandelbrot("mandelbrot"),
        MatrixMul("mxm"),
        MatrixTranspose("mt"),
        MatrixVector("mxv"),
        Montecarlo("montecarlo");

        String id;
        BenchmarkID(String id) {
            this.id = id;
        }

        String id() {
            return id;
        }
    }

    static {
        DEFAULT.put(BenchmarkID.Blackscholes, new DefaultCatalog(1, 8192 * 4096));
        DEFAULT.put(BenchmarkID.DFT, new DefaultCatalog(1, 8192));
        DEFAULT.put(BenchmarkID.Mandelbrot, new DefaultCatalog(1, 8192));
        DEFAULT.put(BenchmarkID.MatrixMul, new DefaultCatalog(2, 1024));
        DEFAULT.put(BenchmarkID.MatrixTranspose, new DefaultCatalog(2, 8192));
        DEFAULT.put(BenchmarkID.MatrixVector, new DefaultCatalog(1, 8192 * 2));
        DEFAULT.put(BenchmarkID.Montecarlo, new DefaultCatalog(1, 16777216 * 8));
    }

    private Catalog() {}
}
