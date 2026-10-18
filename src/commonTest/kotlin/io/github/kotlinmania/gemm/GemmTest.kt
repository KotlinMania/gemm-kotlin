// port-lint: tests lib.rs
package io.github.kotlinmania.gemm

import io.github.kotlinmania.half.F16
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class GemmTest {
    @Test
    fun testGemmF16() {
        val mnks =
            listOf(
                Triple(4, 4, 4),
                Triple(63, 2, 10),
                Triple(16, 2, 1),
                Triple(0, 0, 4),
                Triple(16, 1, 1),
                Triple(16, 3, 1),
                Triple(16, 4, 1),
                Triple(16, 1, 2),
                Triple(16, 2, 2),
                Triple(16, 3, 2),
                Triple(16, 4, 2),
                Triple(16, 16, 1),
                Triple(64, 64, 0),
                Triple(64, 64, 4),
                Triple(0, 64, 4),
                Triple(64, 0, 4),
                Triple(8, 16, 1),
                Triple(16, 8, 1),
                Triple(1, 1, 2),
                Triple(63, 1, 10),
                Triple(63, 3, 10),
                Triple(63, 4, 10),
                Triple(1, 63, 10),
                Triple(2, 63, 10),
                Triple(3, 63, 10),
                Triple(4, 63, 10),
            )

        val rng = Random(42)
        for ((m, n, k) in mnks) {
            for (parallelism in listOf(Parallelism.None, Parallelism.Rayon(0))) {
                for (alphaVal in listOf(0.0f, 1.0f, 2.3f)) {
                    for (betaVal in listOf(0.0f, 1.0f, 2.3f)) {
                        for (colmajor in listOf(true, false)) {
                            val alpha = F16.fromF32(alphaVal)
                            val beta = F16.fromF32(betaVal)
                            val aVec = Array(m * k) { F16.fromF32(rng.nextFloat()) }
                            val bVec = Array(k * n) { F16.fromF32(rng.nextFloat()) }
                            val cVec = Array(m * n) { F16.fromF32(rng.nextFloat()) }
                            val dVec = cVec.copyOf()

                            val dstCs = if (colmajor) m.toLong() else 1L
                            val dstRs = if (colmajor) 1L else n.toLong()

                            gemm(
                                m,
                                n,
                                k,
                                cVec.asMutPtr(),
                                dstCs,
                                dstRs,
                                true,
                                aVec.asPtr(),
                                m.toLong(),
                                1L,
                                bVec.asPtr(),
                                k.toLong(),
                                1L,
                                alpha,
                                beta,
                                false,
                                false,
                                false,
                                parallelism,
                            )

                            gemmFallback(
                                m,
                                n,
                                k,
                                dVec.asMutPtr(),
                                dstCs,
                                dstRs,
                                true,
                                aVec.asPtr(),
                                m.toLong(),
                                1L,
                                bVec.asPtr(),
                                k.toLong(),
                                1L,
                                alpha,
                                beta,
                            )

                            val eps = 0.2f
                            for (i in cVec.indices) {
                                val c = cVec[i].toF32()
                                val d = dVec[i].toF32()
                                val diff = abs(c - d)
                                val epsRel = abs(c) * eps
                                val maxEps = maxOf(epsRel, eps)
                                assertTrue(diff <= maxEps + 1e-2f, "Mismatch at $i: c=$c, d=$d, diff=$diff, tol=$maxEps")
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun testGemmF32() {
        setWasmSimd128(true)

        val mnks =
            listOf(
                Triple(63, 2, 10),
                Triple(1, 2, 10),
                Triple(1, 63, 10),
                Triple(64, 64, 4),
                Triple(0, 64, 4),
                Triple(64, 0, 4),
                Triple(0, 0, 4),
                Triple(64, 64, 0),
                Triple(16, 1, 1),
                Triple(16, 2, 1),
                Triple(16, 3, 1),
                Triple(16, 4, 1),
                Triple(16, 1, 2),
                Triple(16, 2, 2),
                Triple(16, 3, 2),
                Triple(16, 4, 2),
                Triple(16, 16, 1),
                Triple(8, 16, 1),
                Triple(16, 8, 1),
                Triple(1, 1, 2),
                Triple(4, 4, 4),
                Triple(63, 1, 10),
                Triple(63, 3, 10),
                Triple(63, 4, 10),
                Triple(2, 63, 10),
                Triple(3, 63, 10),
                Triple(4, 63, 10),
            )

        val rng = Random(42)
        for ((m, n, k) in mnks) {
            for (parallelism in listOf(Parallelism.None, Parallelism.Rayon(0))) {
                for (alpha in listOf(0.0f, 1.0f, 2.3f)) {
                    for (beta in listOf(0.0f, 1.0f, 2.3f)) {
                        for (colmajor in listOf(true, false)) {
                            val aVec = FloatArray(m * k) { rng.nextFloat() }
                            val bVec = FloatArray(k * n) { rng.nextFloat() }
                            val cVec = FloatArray(m * n) { rng.nextFloat() }
                            val dVec = cVec.copyOf()

                            val dstCs = if (colmajor) m.toLong() else 1L
                            val dstRs = if (colmajor) 1L else n.toLong()

                            gemm(
                                m,
                                n,
                                k,
                                cVec.asMutPtr(),
                                dstCs,
                                dstRs,
                                true,
                                aVec.asPtr(),
                                m.toLong(),
                                1L,
                                bVec.asPtr(),
                                k.toLong(),
                                1L,
                                alpha,
                                beta,
                                false,
                                false,
                                false,
                                parallelism,
                            )

                            gemmFallback(
                                m,
                                n,
                                k,
                                dVec.asMutPtr(),
                                dstCs,
                                dstRs,
                                true,
                                aVec.asPtr(),
                                m.toLong(),
                                1L,
                                bVec.asPtr(),
                                k.toLong(),
                                1L,
                                alpha,
                                beta,
                            )

                            for (i in cVec.indices) {
                                val c = cVec[i]
                                val d = dVec[i]
                                assertTrue(abs(c - d) <= 1e-3f, "Mismatch at $i: c=$c, d=$d")
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun testGemmF64() {
        setWasmSimd128(true)

        val mnks =
            listOf(
                Triple(63, 2, 10),
                Triple(1, 2, 10),
                Triple(1, 63, 10),
                Triple(64, 64, 4),
                Triple(0, 64, 4),
                Triple(64, 0, 4),
                Triple(0, 0, 4),
                Triple(64, 64, 0),
                Triple(16, 1, 1),
                Triple(16, 2, 1),
                Triple(16, 3, 1),
                Triple(16, 4, 1),
                Triple(16, 1, 2),
                Triple(16, 2, 2),
                Triple(16, 3, 2),
                Triple(16, 4, 2),
                Triple(16, 16, 1),
                Triple(8, 16, 1),
                Triple(16, 8, 1),
                Triple(1, 1, 2),
                Triple(4, 4, 4),
                Triple(63, 1, 10),
                Triple(63, 3, 10),
                Triple(63, 4, 10),
                Triple(2, 63, 10),
                Triple(3, 63, 10),
                Triple(4, 63, 10),
            )

        val rng = Random(42)
        for ((m, n, k) in mnks) {
            for (parallelism in listOf(Parallelism.None, Parallelism.Rayon(0))) {
                for (alpha in listOf(0.0, 1.0, 2.3)) {
                    for (beta in listOf(0.0, 1.0, 2.3)) {
                        for (colmajor in listOf(true, false)) {
                            val aVec = DoubleArray(m * k) { rng.nextDouble() }
                            val bVec = DoubleArray(k * n) { rng.nextDouble() }
                            val cVec = DoubleArray(m * n) { rng.nextDouble() }
                            val dVec = cVec.copyOf()

                            val dstCs = if (colmajor) m.toLong() else 1L
                            val dstRs = if (colmajor) 1L else n.toLong()

                            gemm(
                                m,
                                n,
                                k,
                                cVec.asMutPtr(),
                                dstCs,
                                dstRs,
                                true,
                                aVec.asPtr(),
                                m.toLong(),
                                1L,
                                bVec.asPtr(),
                                k.toLong(),
                                1L,
                                alpha,
                                beta,
                                false,
                                false,
                                false,
                                parallelism,
                            )

                            gemmFallback(
                                m,
                                n,
                                k,
                                dVec.asMutPtr(),
                                dstCs,
                                dstRs,
                                true,
                                aVec.asPtr(),
                                m.toLong(),
                                1L,
                                bVec.asPtr(),
                                k.toLong(),
                                1L,
                                alpha,
                                beta,
                            )

                            for (i in cVec.indices) {
                                val c = cVec[i]
                                val d = dVec[i]
                                assertTrue(abs(c - d) <= 1e-5, "Mismatch at $i: c=$c, d=$d")
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun testGemmCplx32() {
        val mnks =
            listOf(
                Triple(4, 4, 4),
                Triple(0, 64, 4),
                Triple(64, 0, 4),
                Triple(0, 0, 4),
                Triple(64, 64, 4),
                Triple(64, 64, 0),
                Triple(6, 3, 1),
                Triple(1, 1, 2),
                Triple(16, 1, 1),
                Triple(16, 2, 1),
                Triple(16, 3, 1),
                Triple(16, 4, 1),
                Triple(16, 1, 2),
                Triple(16, 2, 2),
                Triple(16, 3, 2),
                Triple(16, 4, 2),
                Triple(16, 16, 1),
                Triple(8, 16, 1),
                Triple(16, 8, 1),
                Triple(63, 1, 10),
                Triple(63, 2, 10),
                Triple(63, 3, 10),
                Triple(63, 4, 10),
                Triple(1, 63, 10),
                Triple(2, 63, 10),
                Triple(3, 63, 10),
                Triple(4, 63, 10),
            )

        val rng = Random(42)
        val zero = Complex32.ZERO
        val one = Complex32.ONE
        val arbitrary = Complex32.new(2.3f, 4.1f)

        for ((m, n, k) in mnks) {
            for (alpha in listOf(zero, one, arbitrary)) {
                for (beta in listOf(zero, one, arbitrary)) {
                    for (conjDst in listOf(false, true)) {
                        for (conjLhs in listOf(false, true)) {
                            for (conjRhs in listOf(false, true)) {
                                for (colmajor in listOf(true, false)) {
                                    val aVec = Array(m * k) { Complex32.new(rng.nextFloat(), rng.nextFloat()) }
                                    val bVec = Array(k * n) { Complex32.new(rng.nextFloat(), rng.nextFloat()) }
                                    val cVec = Array(m * n) { Complex32.new(rng.nextFloat(), rng.nextFloat()) }
                                    val dVec = cVec.copyOf()

                                    val dstCs = if (colmajor) m.toLong() else 1L
                                    val dstRs = if (colmajor) 1L else n.toLong()

                                    gemm(
                                        m,
                                        n,
                                        k,
                                        cVec.asMutPtr(),
                                        dstCs,
                                        dstRs,
                                        true,
                                        aVec.asPtr(),
                                        m.toLong(),
                                        1L,
                                        bVec.asPtr(),
                                        k.toLong(),
                                        1L,
                                        alpha,
                                        beta,
                                        conjDst,
                                        conjLhs,
                                        conjRhs,
                                        Parallelism.None,
                                    )

                                    gemmCplxFallback(
                                        m,
                                        n,
                                        k,
                                        dVec.asMutPtr(),
                                        dstCs,
                                        dstRs,
                                        true,
                                        aVec.asPtr(),
                                        m.toLong(),
                                        1L,
                                        bVec.asPtr(),
                                        k.toLong(),
                                        1L,
                                        alpha,
                                        beta,
                                        conjDst,
                                        conjLhs,
                                        conjRhs,
                                    )

                                    for (i in cVec.indices) {
                                        val c = cVec[i]
                                        val d = dVec[i]
                                        assertTrue(abs(c.re - d.re) <= 1e-3f, "Mismatch re at $i: c=$c, d=$d")
                                        assertTrue(abs(c.im - d.im) <= 1e-3f, "Mismatch im at $i: c=$c, d=$d")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun testGemmCplx64() {
        val mnks =
            listOf(
                Triple(4, 4, 4),
                Triple(0, 64, 4),
                Triple(64, 0, 4),
                Triple(0, 0, 4),
                Triple(64, 64, 4),
                Triple(64, 64, 0),
                Triple(6, 3, 1),
                Triple(1, 1, 2),
                Triple(16, 1, 1),
                Triple(16, 2, 1),
                Triple(16, 3, 1),
                Triple(16, 4, 1),
                Triple(16, 1, 2),
                Triple(16, 2, 2),
                Triple(16, 3, 2),
                Triple(16, 4, 2),
                Triple(16, 16, 1),
                Triple(8, 16, 1),
                Triple(16, 8, 1),
                Triple(63, 1, 10),
                Triple(63, 2, 10),
                Triple(63, 3, 10),
                Triple(63, 4, 10),
                Triple(1, 63, 10),
                Triple(2, 63, 10),
                Triple(3, 63, 10),
                Triple(4, 63, 10),
            )

        val rng = Random(42)
        val zero = Complex64.ZERO
        val one = Complex64.ONE
        val arbitrary = Complex64.new(2.3, 4.1)

        for ((m, n, k) in mnks) {
            for (alpha in listOf(zero, one, arbitrary)) {
                for (beta in listOf(zero, one, arbitrary)) {
                    for (conjDst in listOf(false, true)) {
                        for (conjLhs in listOf(false, true)) {
                            for (conjRhs in listOf(false, true)) {
                                for (colmajor in listOf(true, false)) {
                                    val aVec = Array(m * k) { Complex64.new(rng.nextDouble(), rng.nextDouble()) }
                                    val bVec = Array(k * n) { Complex64.new(rng.nextDouble(), rng.nextDouble()) }
                                    val cVec = Array(m * n) { Complex64.new(rng.nextDouble(), rng.nextDouble()) }
                                    val dVec = cVec.copyOf()

                                    val dstCs = if (colmajor) m.toLong() else 1L
                                    val dstRs = if (colmajor) 1L else n.toLong()

                                    gemm(
                                        m,
                                        n,
                                        k,
                                        cVec.asMutPtr(),
                                        dstCs,
                                        dstRs,
                                        true,
                                        aVec.asPtr(),
                                        m.toLong(),
                                        1L,
                                        bVec.asPtr(),
                                        k.toLong(),
                                        1L,
                                        alpha,
                                        beta,
                                        conjDst,
                                        conjLhs,
                                        conjRhs,
                                        Parallelism.None,
                                    )

                                    gemmCplxFallback(
                                        m,
                                        n,
                                        k,
                                        dVec.asMutPtr(),
                                        dstCs,
                                        dstRs,
                                        true,
                                        aVec.asPtr(),
                                        m.toLong(),
                                        1L,
                                        bVec.asPtr(),
                                        k.toLong(),
                                        1L,
                                        alpha,
                                        beta,
                                        conjDst,
                                        conjLhs,
                                        conjRhs,
                                    )

                                    for (i in cVec.indices) {
                                        val c = cVec[i]
                                        val d = dVec[i]
                                        assertTrue(abs(c.re - d.re) <= 1e-5, "Mismatch re at $i: c=$c, d=$d")
                                        assertTrue(abs(c.im - d.im) <= 1e-5, "Mismatch im at $i: c=$c, d=$d")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
