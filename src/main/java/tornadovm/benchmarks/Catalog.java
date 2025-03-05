package tornadovm.benchmarks;

import java.util.HashMap;
import java.util.Map;

public class Catalog {

    public static final Map<String, DefaultCatalog> DEFAULT = new HashMap<>();

    public record DefaultCatalog(int dimensions, int size) {}

    static {
        DEFAULT.put("dft", new DefaultCatalog(1, 8192));
        DEFAULT.put("mandelbrot", new DefaultCatalog(1, 8192));
        DEFAULT.put("mxm", new DefaultCatalog(2, 1024));
        DEFAULT.put("mt", new DefaultCatalog(2, 8192));
        DEFAULT.put("mxv", new DefaultCatalog(1, 8192 * 2));
        DEFAULT.put("montecarlo", new DefaultCatalog(1, 16777216 * 8));
    }

    private Catalog() {}
}
