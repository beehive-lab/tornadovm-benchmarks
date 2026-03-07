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
 * Validates every Java backend of {@link Silu} against its sequential reference.
 *
 * <p>Key fixes under test:
 * <ul>
 *   <li>{@code computeWithParallelVectorAPI}: missing {@code vA} multiplication and
 *       inaccurate 2nd-order Taylor approximation for {@code exp(-x)}.</li>
 * </ul>
 */
public class SiluTest {

    private static final int SIZE = 256;

    private Silu silu;

    @Before
    public void setUp() {
        silu = new Silu(SIZE);
        // Establish the sequential reference in shbRef.
        silu.computeSequential();
    }

    @Test
    public void testJavaStreams() {
        silu.resetOutputs();
        silu.computeWithJavaStreams();
        assertTrue("Silu streams result does not match sequential reference", silu.isResultCorrect());
    }

    @Test
    public void testJavaThreads() throws InterruptedException {
        silu.resetOutputs();
        silu.computeWithJavaThreads();
        assertTrue("Silu threads result does not match sequential reference", silu.isResultCorrect());
    }

    @Test
    public void testJavaThreadsReusing() throws InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        try {
            silu.resetOutputs();
            silu.computeWithJavaThreadsReusing(executor);
            assertTrue("Silu thread-pool result does not match sequential reference", silu.isResultCorrect());
        } finally {
            executor.shutdown();
        }
    }

    @Test
    public void testVectorAPI() {
        silu.resetOutputs();
        silu.computeWithParallelVectorAPI();
        assertTrue("Silu VectorAPI result does not match sequential reference", silu.isResultCorrect());
    }
}
