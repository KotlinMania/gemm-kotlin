// port-lint: source gemm.rs
package io.github.kotlinmania.gemm

import io.github.kotlinmania.half.F16
import kotlin.math.abs

public typealias f16 = F16
public typealias c32 = Complex32
public typealias c64 = Complex64

public fun <T : Any> gemmDispatch(
    m: Int,
    n: Int,
    k: Int,
    dst: Ptr<T>,
    dstCs: Long,
    dstRs: Long,
    readDst: Boolean,
    lhs: Ptr<T>,
    lhsCs: Long,
    lhsRs: Long,
    rhs: Ptr<T>,
    rhsCs: Long,
    rhsRs: Long,
    alpha: T,
    beta: T,
    conjDst: Boolean,
    conjLhs: Boolean,
    conjRhs: Boolean,
    parallelism: Parallelism,
) {
    when (parallelism) {
        is Parallelism.None -> {}
        is Parallelism.Rayon -> {}
    }

    if (m == 0 || n == 0) return

    if (dst is FloatArrayPtr && lhs is FloatArrayPtr && rhs is FloatArrayPtr && alpha is Float && beta is Float) {
        val dstArray = dst.array
        val lhsArray = lhs.array
        val rhsArray = rhs.array
        val dstOffset = dst.offset
        val lhsOffset = lhs.offset
        val rhsOffset = rhs.offset
        for (row in 0 until m) {
            val rowLhs = lhsOffset + row * lhsRs
            val rowDst = dstOffset + row * dstRs
            for (col in 0 until n) {
                var accum = 0.0f
                val colRhs = rhsOffset + col * rhsCs
                for (depth in 0 until k) {
                    val l = lhsArray[(rowLhs + depth * lhsCs).toInt()]
                    val r = rhsArray[(depth * rhsRs + colRhs).toInt()]
                    accum += l * r
                }
                accum *= beta
                val idx = (rowDst + col * dstCs).toInt()
                if (readDst) {
                    accum += alpha * dstArray[idx]
                }
                dstArray[idx] = accum
            }
        }
        return
    }

    if (dst is DoubleArrayPtr && lhs is DoubleArrayPtr && rhs is DoubleArrayPtr && alpha is Double && beta is Double) {
        val dstArray = dst.array
        val lhsArray = lhs.array
        val rhsArray = rhs.array
        val dstOffset = dst.offset
        val lhsOffset = lhs.offset
        val rhsOffset = rhs.offset
        for (row in 0 until m) {
            val rowLhs = lhsOffset + row * lhsRs
            val rowDst = dstOffset + row * dstRs
            for (col in 0 until n) {
                var accum = 0.0
                val colRhs = rhsOffset + col * rhsCs
                for (depth in 0 until k) {
                    val l = lhsArray[(rowLhs + depth * lhsCs).toInt()]
                    val r = rhsArray[(depth * rhsRs + colRhs).toInt()]
                    accum += l * r
                }
                accum *= beta
                val idx = (rowDst + col * dstCs).toInt()
                if (readDst) {
                    accum += alpha * dstArray[idx]
                }
                dstArray[idx] = accum
            }
        }
        return
    }

    when (alpha) {
        is Float -> {
            val a = alpha as Float
            val b = beta as Float
            for (row in 0 until m) {
                for (col in 0 until n) {
                    var accum = 0.0f
                    for (depth in 0 until k) {
                        val l = lhs.get((row * lhsRs + depth * lhsCs).toInt()) as Float
                        val r = rhs.get((depth * rhsRs + col * rhsCs).toInt()) as Float
                        accum += l * r
                    }
                    accum *= b
                    val target = dst.wrappingOffset(row * dstRs + col * dstCs)
                    if (readDst) {
                        accum += a * (target.read() as Float)
                    }
                    target.writeAny(accum)
                }
            }
        }

        is Double -> {
            val a = alpha as Double
            val b = beta as Double
            for (row in 0 until m) {
                for (col in 0 until n) {
                    var accum = 0.0
                    for (depth in 0 until k) {
                        val l = lhs.get((row * lhsRs + depth * lhsCs).toInt()) as Double
                        val r = rhs.get((depth * rhsRs + col * rhsCs).toInt()) as Double
                        accum += l * r
                    }
                    accum *= b
                    val target = dst.wrappingOffset(row * dstRs + col * dstCs)
                    if (readDst) {
                        accum += a * (target.read() as Double)
                    }
                    target.writeAny(accum)
                }
            }
        }

        is F16 -> {
            val a = alpha as F16
            val b = beta as F16
            for (row in 0 until m) {
                for (col in 0 until n) {
                    var accum = F16.ZERO
                    for (depth in 0 until k) {
                        val l = lhs.get((row * lhsRs + depth * lhsCs).toInt()) as F16
                        val r = rhs.get((depth * rhsRs + col * rhsCs).toInt()) as F16
                        accum = accum + (l * r)
                    }
                    accum = accum * b
                    val target = dst.wrappingOffset(row * dstRs + col * dstCs)
                    if (readDst) {
                        accum = accum + (a * (target.read() as F16))
                    }
                    target.writeAny(accum)
                }
            }
        }

        is Complex32 -> {
            val a = alpha as Complex32
            val b = beta as Complex32
            for (row in 0 until m) {
                for (col in 0 until n) {
                    var accum = Complex32.ZERO
                    for (depth in 0 until k) {
                        val lRaw = lhs.get((row * lhsRs + depth * lhsCs).toInt()) as Complex32
                        val rRaw = rhs.get((depth * rhsRs + col * rhsCs).toInt()) as Complex32
                        val l = if (conjLhs) lRaw.conj() else lRaw
                        val r = if (conjRhs) rRaw.conj() else rRaw
                        accum = accum + (l * r)
                    }
                    accum = accum * b
                    val target = dst.wrappingOffset(row * dstRs + col * dstCs)
                    if (readDst) {
                        val dRaw = target.read() as Complex32
                        val d = if (conjDst) dRaw.conj() else dRaw
                        accum = accum + (a * d)
                    }
                    target.writeAny(accum)
                }
            }
        }

        is Complex64 -> {
            val a = alpha as Complex64
            val b = beta as Complex64
            for (row in 0 until m) {
                for (col in 0 until n) {
                    var accum = Complex64.ZERO
                    for (depth in 0 until k) {
                        val lRaw = lhs.get((row * lhsRs + depth * lhsCs).toInt()) as Complex64
                        val rRaw = rhs.get((depth * rhsRs + col * rhsCs).toInt()) as Complex64
                        val l = if (conjLhs) lRaw.conj() else lRaw
                        val r = if (conjRhs) rRaw.conj() else rRaw
                        accum = accum + (l * r)
                    }
                    accum = accum * b
                    val target = dst.wrappingOffset(row * dstRs + col * dstCs)
                    if (readDst) {
                        val dRaw = target.read() as Complex64
                        val d = if (conjDst) dRaw.conj() else dRaw
                        accum = accum + (a * d)
                    }
                    target.writeAny(accum)
                }
            }
        }

        else -> {
            error("Unsupported gemm type: ${alpha::class}")
        }
    }
}

/**
 * Computes destination := alpha * destination + beta * lhs * rhs.
 *
 * Throws an error if type `T` is not supported.
 */
public fun <T : Any> gemm(
    m: Int,
    n: Int,
    k: Int,
    dst: Ptr<T>,
    dstCs: Long,
    dstRs: Long,
    readDst: Boolean,
    lhs: Ptr<T>,
    lhsCs: Long,
    lhsRs: Long,
    rhs: Ptr<T>,
    rhsCs: Long,
    rhsRs: Long,
    alpha: T,
    beta: T,
    conjDst: Boolean,
    conjLhs: Boolean,
    conjRhs: Boolean,
    parallelism: Parallelism,
) {
    val doTranspose = abs(dstCs) < abs(dstRs)

    var curM = m
    var curN = n
    var curDstCs = dstCs
    var curDstRs = dstRs
    var curLhs = lhs
    var curLhsCs = lhsCs
    var curLhsRs = lhsRs
    var curRhs = rhs
    var curRhsCs = rhsCs
    var curRhsRs = rhsRs
    var curConjLhs = conjLhs
    var curConjRhs = conjRhs
    var curDst = dst

    if (doTranspose) {
        curM = n
        curN = m
        curDstCs = dstRs
        curDstRs = dstCs
        curLhs = rhs
        curLhsCs = rhsRs
        curLhsRs = rhsCs
        curRhs = lhs
        curRhsCs = lhsRs
        curRhsRs = lhsCs
        curConjLhs = conjRhs
        curConjRhs = conjLhs
    }

    if (curDstRs < 0 && curM > 0) {
        curDst = curDst.wrappingOffset((curM - 1).toLong() * curDstRs)
        curDstRs = -curDstRs
        curLhs = curLhs.wrappingOffset((curM - 1).toLong() * curLhsRs)
        curLhsRs = -curLhsRs
    }

    if (curDstCs < 0 && curN > 0) {
        curDst = curDst.wrappingOffset((curN - 1).toLong() * curDstCs)
        curDstCs = -curDstCs
        curRhs = curRhs.wrappingOffset((curN - 1).toLong() * curRhsCs)
        curRhsCs = -curRhsCs
    }

    if (curLhsCs < 0 && k > 0) {
        curLhs = curLhs.wrappingOffset((k - 1).toLong() * curLhsCs)
        curLhsCs = -curLhsCs
        curRhs = curRhs.wrappingOffset((k - 1).toLong() * curRhsRs)
        curRhsRs = -curRhsRs
    }

    gemmDispatch(
        curM,
        curN,
        k,
        curDst,
        curDstCs,
        curDstRs,
        readDst,
        curLhs,
        curLhsCs,
        curLhsRs,
        curRhs,
        curRhsCs,
        curRhsRs,
        alpha,
        beta,
        conjDst,
        curConjLhs,
        curConjRhs,
        parallelism,
    )
}

public fun <T : Any> gemm(
    m: Int,
    n: Int,
    k: Int,
    dst: Ptr<T>,
    dstCs: Int,
    dstRs: Int,
    readDst: Boolean,
    lhs: Ptr<T>,
    lhsCs: Int,
    lhsRs: Int,
    rhs: Ptr<T>,
    rhsCs: Int,
    rhsRs: Int,
    alpha: T,
    beta: T,
    conjDst: Boolean,
    conjLhs: Boolean,
    conjRhs: Boolean,
    parallelism: Parallelism,
) {
    gemm(
        m,
        n,
        k,
        dst,
        dstCs.toLong(),
        dstRs.toLong(),
        readDst,
        lhs,
        lhsCs.toLong(),
        lhsRs.toLong(),
        rhs,
        rhsCs.toLong(),
        rhsRs.toLong(),
        alpha,
        beta,
        conjDst,
        conjLhs,
        conjRhs,
        parallelism,
    )
}

public fun <T : Any> gemmFallback(
    m: Int,
    n: Int,
    k: Int,
    dst: Ptr<T>,
    dstCs: Long,
    dstRs: Long,
    readDst: Boolean,
    lhs: Ptr<T>,
    lhsCs: Long,
    lhsRs: Long,
    rhs: Ptr<T>,
    rhsCs: Long,
    rhsRs: Long,
    alpha: T,
    beta: T,
) {
    when (alpha) {
        is Float -> {
            val a = alpha as Float
            val b = beta as Float
            for (row in 0 until m) {
                for (col in 0 until n) {
                    var accum = 0.0f
                    for (depth in 0 until k) {
                        val l = lhs.wrappingOffset(row * lhsRs + depth * lhsCs).read() as Float
                        val r = rhs.wrappingOffset(depth * rhsRs + col * rhsCs).read() as Float
                        accum += l * r
                    }
                    accum *= b
                    val target = dst.wrappingOffset(row * dstRs + col * dstCs)
                    if (readDst) {
                        accum += a * (target.read() as Float)
                    }
                    target.writeAny(accum)
                }
            }
        }

        is Double -> {
            val a = alpha as Double
            val b = beta as Double
            for (row in 0 until m) {
                for (col in 0 until n) {
                    var accum = 0.0
                    for (depth in 0 until k) {
                        val l = lhs.wrappingOffset(row * lhsRs + depth * lhsCs).read() as Double
                        val r = rhs.wrappingOffset(depth * rhsRs + col * rhsCs).read() as Double
                        accum += l * r
                    }
                    accum *= b
                    val target = dst.wrappingOffset(row * dstRs + col * dstCs)
                    if (readDst) {
                        accum += a * (target.read() as Double)
                    }
                    target.writeAny(accum)
                }
            }
        }

        is F16 -> {
            val a = alpha as F16
            val b = beta as F16
            for (row in 0 until m) {
                for (col in 0 until n) {
                    var accum = F16.ZERO
                    for (depth in 0 until k) {
                        val l = lhs.wrappingOffset(row * lhsRs + depth * lhsCs).read() as F16
                        val r = rhs.wrappingOffset(depth * rhsRs + col * rhsCs).read() as F16
                        accum = accum + (l * r)
                    }
                    accum = accum * b
                    val target = dst.wrappingOffset(row * dstRs + col * dstCs)
                    if (readDst) {
                        accum = accum + (a * (target.read() as F16))
                    }
                    target.writeAny(accum)
                }
            }
        }

        else -> {
            error("Unsupported gemm fallback type: ${alpha::class}")
        }
    }
}

public fun <T : Any> gemmFallback(
    m: Int,
    n: Int,
    k: Int,
    dst: Ptr<T>,
    dstCs: Int,
    dstRs: Int,
    readDst: Boolean,
    lhs: Ptr<T>,
    lhsCs: Int,
    lhsRs: Int,
    rhs: Ptr<T>,
    rhsCs: Int,
    rhsRs: Int,
    alpha: T,
    beta: T,
) {
    gemmFallback(
        m,
        n,
        k,
        dst,
        dstCs.toLong(),
        dstRs.toLong(),
        readDst,
        lhs,
        lhsCs.toLong(),
        lhsRs.toLong(),
        rhs,
        rhsCs.toLong(),
        rhsRs.toLong(),
        alpha,
        beta,
    )
}

public fun <T : Any> gemmCplxFallback(
    m: Int,
    n: Int,
    k: Int,
    dst: Ptr<T>,
    dstCs: Long,
    dstRs: Long,
    readDst: Boolean,
    lhs: Ptr<T>,
    lhsCs: Long,
    lhsRs: Long,
    rhs: Ptr<T>,
    rhsCs: Long,
    rhsRs: Long,
    alpha: T,
    beta: T,
    conjDst: Boolean,
    conjLhs: Boolean,
    conjRhs: Boolean,
) {
    when (alpha) {
        is Complex32 -> {
            val a = alpha as Complex32
            val b = beta as Complex32
            for (row in 0 until m) {
                for (col in 0 until n) {
                    var accum = Complex32.ZERO
                    for (depth in 0 until k) {
                        val lRaw = lhs.wrappingOffset(row * lhsRs + depth * lhsCs).read() as Complex32
                        val rRaw = rhs.wrappingOffset(depth * rhsRs + col * rhsCs).read() as Complex32
                        val l = if (conjLhs) lRaw.conj() else lRaw
                        val r = if (conjRhs) rRaw.conj() else rRaw
                        accum = accum + (l * r)
                    }
                    accum = accum * b
                    val target = dst.wrappingOffset(row * dstRs + col * dstCs)
                    if (readDst) {
                        val dRaw = target.read() as Complex32
                        val d = if (conjDst) dRaw.conj() else dRaw
                        accum = accum + (a * d)
                    }
                    target.writeAny(accum)
                }
            }
        }

        is Complex64 -> {
            val a = alpha as Complex64
            val b = beta as Complex64
            for (row in 0 until m) {
                for (col in 0 until n) {
                    var accum = Complex64.ZERO
                    for (depth in 0 until k) {
                        val lRaw = lhs.wrappingOffset(row * lhsRs + depth * lhsCs).read() as Complex64
                        val rRaw = rhs.wrappingOffset(depth * rhsRs + col * rhsCs).read() as Complex64
                        val l = if (conjLhs) lRaw.conj() else lRaw
                        val r = if (conjRhs) rRaw.conj() else rRaw
                        accum = accum + (l * r)
                    }
                    accum = accum * b
                    val target = dst.wrappingOffset(row * dstRs + col * dstCs)
                    if (readDst) {
                        val dRaw = target.read() as Complex64
                        val d = if (conjDst) dRaw.conj() else dRaw
                        accum = accum + (a * d)
                    }
                    target.writeAny(accum)
                }
            }
        }

        else -> {
            error("Unsupported gemm complex fallback type: ${alpha::class}")
        }
    }
}

public fun <T : Any> gemmCplxFallback(
    m: Int,
    n: Int,
    k: Int,
    dst: Ptr<T>,
    dstCs: Int,
    dstRs: Int,
    readDst: Boolean,
    lhs: Ptr<T>,
    lhsCs: Int,
    lhsRs: Int,
    rhs: Ptr<T>,
    rhsCs: Int,
    rhsRs: Int,
    alpha: T,
    beta: T,
    conjDst: Boolean,
    conjLhs: Boolean,
    conjRhs: Boolean,
) {
    gemmCplxFallback(
        m,
        n,
        k,
        dst,
        dstCs.toLong(),
        dstRs.toLong(),
        readDst,
        lhs,
        lhsCs.toLong(),
        lhsRs.toLong(),
        rhs,
        rhsCs.toLong(),
        rhsRs.toLong(),
        alpha,
        beta,
        conjDst,
        conjLhs,
        conjRhs,
    )
}
