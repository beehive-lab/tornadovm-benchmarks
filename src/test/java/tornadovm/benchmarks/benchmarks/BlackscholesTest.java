/*
 * Copyright (c) 2026, APT Group, Department of Computer Science,
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
package tornadovm.benchmarks.benchmarks;

import org.junit.Before;
import org.junit.Test;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.junit.Assert.assertTrue;

/**
 * Validates every Java backend of {@link Blackscholes} against its sequential reference.
 *
 * <p>Key fixes under test:
 * <ul>
 *   <li>{@code computeWithParallelVectorAPI}: was a scalar loop with no {@code FloatVector}
 *       operations; replaced with a VectorAPI implementation using chunked memory loads
 *       and stores.</li>
 *   <li>(Previous session) {@code computeWithParallelVectorAPI} was writing results to
 *       {@code callResultRef/putResultRef} instead of {@code callResult/putResult}.</li>
 * </ul>
 */
public class BlackscholesTest {

    private static final int SIZE = 1024;

    private Blackscholes blackscholes;

    @Before
    public void setUp() {
        blackscholes = new Blackscholes(SIZE);
        // Establish the sequential reference in callResultRef / putResultRef.
        blackscholes.computeSequential();
    }

    @Test
    public void testJavaStreams() {
        blackscholes.resetOutputs();
        blackscholes.computeWithJavaStreams();
        assertTrue("Blackscholes streams result does not match sequential reference",
                blackscholes.isResultCorrect());
    }

    @Test
    public void testJavaThreads() throws InterruptedException {
        blackscholes.resetOutputs();
        blackscholes.computeWithJavaThreads();
        assertTrue("Blackscholes threads result does not match sequential reference",
                blackscholes.isResultCorrect());
    }

    @Test
    public void testJavaThreadsReusing() throws InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        try {
            blackscholes.resetOutputs();
            blackscholes.computeWithJavaThreadsReusing(executor);
            assertTrue("Blackscholes thread-pool result does not match sequential reference",
                    blackscholes.isResultCorrect());
        } finally {
            executor.shutdown();
        }
    }

    @Test
    public void testVectorAPI() {
        blackscholes.resetOutputs();
        blackscholes.computeWithParallelVectorAPI();
        assertTrue("Blackscholes VectorAPI result does not match sequential reference",
                blackscholes.isResultCorrect());
    }
}
