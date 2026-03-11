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
 * Validates every Java backend of {@link DFT} against its sequential reference.
 *
 * <p>Key fix under test: {@code resetOutputs} was zeroing {@code outrealRef/outimagRef}
 * (the sequential reference arrays) instead of {@code outreal/outimag} (the output
 * arrays), causing all parallel backends to compare against zeroed data.
 *
 * <p>Small size (128) is used because DFT is O(N²).
 */
public class DFTTest {

    private static final int SIZE = 128;

    private DFT dft;

    @Before
    public void setUp() {
        dft = new DFT(SIZE);
        // Establish the sequential reference in outrealRef / outimagRef.
        dft.computeSequential();
    }

    @Test
    public void testJavaStreams() {
        dft.resetOutputs();
        dft.computeWithJavaStreams();
        assertTrue("DFT streams result does not match sequential reference", dft.isResultCorrect());
    }

    @Test
    public void testJavaThreads() throws InterruptedException {
        dft.resetOutputs();
        dft.computeWithJavaThreads();
        assertTrue("DFT threads result does not match sequential reference", dft.isResultCorrect());
    }

    @Test
    public void testJavaThreadsReusing() throws InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        try {
            dft.resetOutputs();
            dft.computeWithJavaThreadsReusing(executor);
            assertTrue("DFT thread-pool result does not match sequential reference", dft.isResultCorrect());
        } finally {
            executor.shutdown();
        }
    }

    @Test
    public void testVectorAPI() {
        dft.resetOutputs();
        dft.computeWithParallelVectorAPI();
        assertTrue("DFT VectorAPI result does not match sequential reference", dft.isResultCorrect());
    }
}
