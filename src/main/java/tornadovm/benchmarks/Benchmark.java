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

import org.openjdk.jmh.runner.RunnerException;

public abstract class Benchmark {

    abstract int getSize();

    abstract void runWithJMH() throws RunnerException;

    abstract void runTestAll(int size, Option option) throws InterruptedException;

    abstract String getName();

    abstract String printSize();

    public void run(String[] args) throws InterruptedException {
        System.out.println("[INFO] " + getName());
        final int size = getSize();
        System.out.println("[INFO] " +  getName() + " size: " + printSize());

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
