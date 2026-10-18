package io.github.kotlinmania.gemm

/**
 * 32-bit floating point complex number representation.
 */
public data class Complex32(
    public val re: Float = 0.0f,
    public val im: Float = 0.0f,
) {
    public fun conj(): Complex32 = Complex32(re, -im)

    public operator fun unaryMinus(): Complex32 = Complex32(-re, -im)

    public operator fun plus(rhs: Complex32): Complex32 =
        Complex32(re + rhs.re, im + rhs.im)

    public operator fun minus(rhs: Complex32): Complex32 =
        Complex32(re - rhs.re, im - rhs.im)

    public operator fun times(rhs: Complex32): Complex32 =
        Complex32(re * rhs.re - im * rhs.im, re * rhs.im + im * rhs.re)

    public operator fun times(rhs: Float): Complex32 =
        Complex32(re * rhs, im * rhs)

    public companion object {
        public fun new(re: Float, im: Float): Complex32 = Complex32(re, im)

        public val ZERO: Complex32 = Complex32(0.0f, 0.0f)
        public val ONE: Complex32 = Complex32(1.0f, 0.0f)
    }
}

/**
 * 64-bit floating point complex number representation.
 */
public data class Complex64(
    public val re: Double = 0.0,
    public val im: Double = 0.0,
) {
    public fun conj(): Complex64 = Complex64(re, -im)

    public operator fun unaryMinus(): Complex64 = Complex64(-re, -im)

    public operator fun plus(rhs: Complex64): Complex64 =
        Complex64(re + rhs.re, im + rhs.im)

    public operator fun minus(rhs: Complex64): Complex64 =
        Complex64(re - rhs.re, im - rhs.im)

    public operator fun times(rhs: Complex64): Complex64 =
        Complex64(re * rhs.re - im * rhs.im, re * rhs.im + im * rhs.re)

    public operator fun times(rhs: Double): Complex64 =
        Complex64(re * rhs, im * rhs)

    public companion object {
        public fun new(re: Double, im: Double): Complex64 = Complex64(re, im)

        public val ZERO: Complex64 = Complex64(0.0, 0.0)
        public val ONE: Complex64 = Complex64(1.0, 0.0)
    }
}
