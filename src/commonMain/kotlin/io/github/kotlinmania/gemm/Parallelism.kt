package io.github.kotlinmania.gemm

/**
 * Controls threading execution for matrix multiplication operations.
 */
public sealed class Parallelism {
    public data object None : Parallelism()

    public data class Rayon(
        public val threads: Int,
    ) : Parallelism()
}
