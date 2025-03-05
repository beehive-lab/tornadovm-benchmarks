package tornadovm.benchmarks;

import java.util.HashMap;
import java.util.Map;

public class Catalog {

    public static final Map<BenchmarkID, DefaultCatalog> DEFAULT = new HashMap<>();

    public record DefaultCatalog(int dimensions, int size) {}

    public enum BenchmarkID {
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
        DEFAULT.put(BenchmarkID.DFT, new DefaultCatalog(1, 8192));
        DEFAULT.put(BenchmarkID.Mandelbrot, new DefaultCatalog(1, 8192));
        DEFAULT.put(BenchmarkID.MatrixMul, new DefaultCatalog(2, 1024));
        DEFAULT.put(BenchmarkID.MatrixTranspose, new DefaultCatalog(2, 8192));
        DEFAULT.put(BenchmarkID.MatrixVector, new DefaultCatalog(1, 8192 * 2));
        DEFAULT.put(BenchmarkID.Montecarlo, new DefaultCatalog(1, 16777216 * 8));
    }

    private Catalog() {}
}
