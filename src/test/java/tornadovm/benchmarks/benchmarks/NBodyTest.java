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
 * Validates every Java backend of {@link NBody} against its sequential reference.
 *
 * <p>Key fix under test: the sequential reference was written to {@code pos/vel} while
 * threads wrote to {@code posRef/velRef}, and {@code resetOutputs()} cleared both —
 * so validate always compared initial values against a result, never two results.
 * After the fix, sequential writes to {@code posRef/velRef} and all parallel backends
 * write to {@code pos/vel}.
 *
 * <p>Small body count (32) keeps the O(N²) test fast. The Jacobi snapshot algorithm
 * used by all backends guarantees identical results up to floating-point ordering
 * differences, well within {@code Config.DELTA = 0.1} for {@code delT = 0.005}.
 */
public class NBodyTest {

    private static final int NUM_BODIES = 32;

    private NBody nbody;

    @Before
    public void setUp() {
        nbody = new NBody(NUM_BODIES);
        // Establish the sequential (Jacobi) reference in posRef/velRef.
        nbody.computeSequential();
    }

    @Test
    public void testJavaStreams() {
        nbody.resetOutputs();
        nbody.computeWithJavaStreams();
        assertTrue("NBody streams result does not match sequential reference", nbody.isResultCorrect());
    }

    @Test
    public void testJavaThreads() throws InterruptedException {
        nbody.resetOutputs();
        nbody.computeWithJavaThreads();
        assertTrue("NBody threads result does not match sequential reference", nbody.isResultCorrect());
    }

    @Test
    public void testJavaThreadsReusing() throws InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        try {
            nbody.resetOutputs();
            nbody.computeWithJavaThreadsReusing(executor);
            assertTrue("NBody thread-pool result does not match sequential reference", nbody.isResultCorrect());
        } finally {
            executor.shutdown();
        }
    }
}
