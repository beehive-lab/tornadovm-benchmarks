package tornadovm.benchmarks;

import org.openjdk.jmh.runner.RunnerException;

public abstract class TornadoBenchmark {

    abstract int getSize();

    abstract void runWithJMH() throws RunnerException;

    abstract void runTestAll(int size, Option option) throws InterruptedException;

    abstract String getName();

    abstract String printSize();

    public void run(String[] args) throws InterruptedException {
        System.out.println("[INFO] " + getName());
        final int size = getSize();
        System.out.println("[INFO] Mandelbrot size: " + printSize());

        Option option = Option.ALL;
        if (args.length > 0) {
            switch (args[0]) {
                case "jmh" -> {
                    try {
                        runWithJMH();
                        return;
                    } catch (Exception e) {
                        System.err.println("An error occurred: " + e.getMessage());
                    }
                }
                case "onlyJavaSeq" -> option = Option.JAVA_SEQ_ONLY;
                case "onlyJava" -> option = Option.JAVA_ONLY;
                case "onlyTornadoVM" -> option = Option.TORNADO_ONLY;
            }
        }
        runTestAll(size, option);
    }
}
