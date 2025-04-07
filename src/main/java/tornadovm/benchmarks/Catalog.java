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

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class Catalog {

    public static final Map<BenchmarkID, DefaultCatalog> DEFAULT = new HashMap<>();

    public static class DefaultCatalog {
        private final int dimensions;
        private final int size;
        private BufferedImage image;

        public DefaultCatalog(int dimensions, int size) {
            this.dimensions = dimensions;
            this.size = size;
        }

        private DefaultCatalog withImage(BufferedImage image) {
            this.image = image;
            return this;
        }

        public int size() {
            return size;
        }

        public int dimensions() {
            return dimensions;
        }

        public BufferedImage image() {
            return image;
        }
    }

    public enum BenchmarkID {
        Blackscholes("Blackscholes"),
        BlurFilter("blurfilter"),
        DFT("dft"),
        JuliaSets("JuliaSets"),
        Mandelbrot("mandelbrot"),
        MatrixMul("mxm"),
        MatrixTranspose("mt"),
        MatrixVector("mxv"),
        Montecarlo("montecarlo"),
        NBody("nbody"),
        Saxpy("saxpy");

        String id;
        BenchmarkID(String id) {
            this.id = id;
        }

        String id() {
            return id;
        }
    }

    private static BufferedImage loadImage(String path) {
        try {
            return ImageIO.read(new File(path));
        } catch (IOException e) {
            throw new RuntimeException("Input file not found: " + path);
        }
    }

    /*
     * Default input sizes for each benchmark.
     * The catalog represents the number of dimensions (1D, 2D or 3D), the input size, and an image associated (if any).
     */
    static {
        DEFAULT.put(BenchmarkID.Blackscholes, new DefaultCatalog(1, 8192 * 4096));
        DEFAULT.put(BenchmarkID.BlurFilter, new DefaultCatalog(2, -1).withImage(loadImage("./images/small.jpg")));
        DEFAULT.put(BenchmarkID.DFT, new DefaultCatalog(1, 8192));
        DEFAULT.put(BenchmarkID.JuliaSets, new DefaultCatalog(2, 4096));
        DEFAULT.put(BenchmarkID.Mandelbrot, new DefaultCatalog(1, 512));
        DEFAULT.put(BenchmarkID.MatrixMul, new DefaultCatalog(2, 1024));
        DEFAULT.put(BenchmarkID.MatrixTranspose, new DefaultCatalog(2, 8192));
        DEFAULT.put(BenchmarkID.MatrixVector, new DefaultCatalog(1, 8192 * 2));
        DEFAULT.put(BenchmarkID.Montecarlo, new DefaultCatalog(1, 16777216 * 8));
        DEFAULT.put(BenchmarkID.NBody, new DefaultCatalog(1, 16384));
        DEFAULT.put(BenchmarkID.Saxpy, new DefaultCatalog(1, 16777216 * 4));
    }

    private Catalog() {}
}
