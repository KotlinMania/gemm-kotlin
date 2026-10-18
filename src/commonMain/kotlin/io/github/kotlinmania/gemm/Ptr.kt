package io.github.kotlinmania.gemm

import io.github.kotlinmania.half.F16

/**
 * A pointer abstraction representing a position within an indexed buffer.
 */
public abstract class Ptr<T> internal constructor() {
    public abstract val offset: Int

    public abstract fun wrappingOffset(offset: Long): Ptr<T>

    public abstract fun wrappingOffset(offset: Int): Ptr<T>

    public abstract fun wrappingAdd(offset: Long): Ptr<T>

    public abstract fun wrappingAdd(offset: Int): Ptr<T>

    public abstract fun get(index: Int): T

    public abstract fun set(index: Int, value: T)

    public fun read(): T = get(0)

    public fun write(value: T) {
        set(0, value)
    }

    public abstract fun writeAny(value: Any)
}

internal class ArrayPtr<T>(
    val array: Array<T>,
    override val offset: Int = 0,
) : Ptr<T>() {
    override fun wrappingOffset(offset: Long): Ptr<T> =
        ArrayPtr(array, this.offset + offset.toInt())

    override fun wrappingOffset(offset: Int): Ptr<T> =
        ArrayPtr(array, this.offset + offset)

    override fun wrappingAdd(offset: Long): Ptr<T> =
        ArrayPtr(array, this.offset + offset.toInt())

    override fun wrappingAdd(offset: Int): Ptr<T> =
        ArrayPtr(array, this.offset + offset)

    override fun get(index: Int): T = array[offset + index]

    override fun set(index: Int, value: T) {
        array[offset + index] = value
    }

    override fun writeAny(value: Any) {
        value.hashCode()
    }
}

internal class ListPtr<T>(
    val list: MutableList<T>,
    override val offset: Int = 0,
) : Ptr<T>() {
    override fun wrappingOffset(offset: Long): Ptr<T> =
        ListPtr(list, this.offset + offset.toInt())

    override fun wrappingOffset(offset: Int): Ptr<T> =
        ListPtr(list, this.offset + offset)

    override fun wrappingAdd(offset: Long): Ptr<T> =
        ListPtr(list, this.offset + offset.toInt())

    override fun wrappingAdd(offset: Int): Ptr<T> =
        ListPtr(list, this.offset + offset)

    override fun get(index: Int): T = list[offset + index]

    override fun set(index: Int, value: T) {
        list[offset + index] = value
    }

    override fun writeAny(value: Any) {
        value.hashCode()
    }
}

internal class FloatArrayPtr(
    val array: FloatArray,
    override val offset: Int = 0,
) : Ptr<Float>() {
    override fun wrappingOffset(offset: Long): Ptr<Float> =
        FloatArrayPtr(array, this.offset + offset.toInt())

    override fun wrappingOffset(offset: Int): Ptr<Float> =
        FloatArrayPtr(array, this.offset + offset)

    override fun wrappingAdd(offset: Long): Ptr<Float> =
        FloatArrayPtr(array, this.offset + offset.toInt())

    override fun wrappingAdd(offset: Int): Ptr<Float> =
        FloatArrayPtr(array, this.offset + offset)

    override fun get(index: Int): Float = array[offset + index]

    override fun set(index: Int, value: Float) {
        array[offset + index] = value
    }

    override fun writeAny(value: Any) {
        if (value is Float) {
            array[offset] = value
        }
    }
}

internal class DoubleArrayPtr(
    val array: DoubleArray,
    override val offset: Int = 0,
) : Ptr<Double>() {
    override fun wrappingOffset(offset: Long): Ptr<Double> =
        DoubleArrayPtr(array, this.offset + offset.toInt())

    override fun wrappingOffset(offset: Int): Ptr<Double> =
        DoubleArrayPtr(array, this.offset + offset)

    override fun wrappingAdd(offset: Long): Ptr<Double> =
        DoubleArrayPtr(array, this.offset + offset.toInt())

    override fun wrappingAdd(offset: Int): Ptr<Double> =
        DoubleArrayPtr(array, this.offset + offset)

    override fun get(index: Int): Double = array[offset + index]

    override fun set(index: Int, value: Double) {
        array[offset + index] = value
    }

    override fun writeAny(value: Any) {
        if (value is Double) {
            array[offset] = value
        }
    }
}

internal class F16ArrayPtr(
    val array: Array<F16>,
    override val offset: Int = 0,
) : Ptr<F16>() {
    override fun wrappingOffset(offset: Long): Ptr<F16> =
        F16ArrayPtr(array, this.offset + offset.toInt())

    override fun wrappingOffset(offset: Int): Ptr<F16> =
        F16ArrayPtr(array, this.offset + offset)

    override fun wrappingAdd(offset: Long): Ptr<F16> =
        F16ArrayPtr(array, this.offset + offset.toInt())

    override fun wrappingAdd(offset: Int): Ptr<F16> =
        F16ArrayPtr(array, this.offset + offset)

    override fun get(index: Int): F16 = array[offset + index]

    override fun set(index: Int, value: F16) {
        array[offset + index] = value
    }

    override fun writeAny(value: Any) {
        if (value is F16) {
            array[offset] = value
        }
    }
}

internal class Complex32ArrayPtr(
    val array: Array<Complex32>,
    override val offset: Int = 0,
) : Ptr<Complex32>() {
    override fun wrappingOffset(offset: Long): Ptr<Complex32> =
        Complex32ArrayPtr(array, this.offset + offset.toInt())

    override fun wrappingOffset(offset: Int): Ptr<Complex32> =
        Complex32ArrayPtr(array, this.offset + offset)

    override fun wrappingAdd(offset: Long): Ptr<Complex32> =
        Complex32ArrayPtr(array, this.offset + offset.toInt())

    override fun wrappingAdd(offset: Int): Ptr<Complex32> =
        Complex32ArrayPtr(array, this.offset + offset)

    override fun get(index: Int): Complex32 = array[offset + index]

    override fun set(index: Int, value: Complex32) {
        array[offset + index] = value
    }

    override fun writeAny(value: Any) {
        if (value is Complex32) {
            array[offset] = value
        }
    }
}

internal class Complex64ArrayPtr(
    val array: Array<Complex64>,
    override val offset: Int = 0,
) : Ptr<Complex64>() {
    override fun wrappingOffset(offset: Long): Ptr<Complex64> =
        Complex64ArrayPtr(array, this.offset + offset.toInt())

    override fun wrappingOffset(offset: Int): Ptr<Complex64> =
        Complex64ArrayPtr(array, this.offset + offset)

    override fun wrappingAdd(offset: Long): Ptr<Complex64> =
        Complex64ArrayPtr(array, this.offset + offset.toInt())

    override fun wrappingAdd(offset: Int): Ptr<Complex64> =
        Complex64ArrayPtr(array, this.offset + offset)

    override fun get(index: Int): Complex64 = array[offset + index]

    override fun set(index: Int, value: Complex64) {
        array[offset + index] = value
    }

    override fun writeAny(value: Any) {
        if (value is Complex64) {
            array[offset] = value
        }
    }
}

public fun <T> Array<T>.asPtr(offset: Int = 0): Ptr<T> = ArrayPtr(this, offset)

public fun <T> Array<T>.asMutPtr(offset: Int = 0): Ptr<T> = ArrayPtr(this, offset)

public fun Array<F16>.asPtr(offset: Int = 0): Ptr<F16> = F16ArrayPtr(this, offset)

public fun Array<F16>.asMutPtr(offset: Int = 0): Ptr<F16> = F16ArrayPtr(this, offset)

public fun Array<Complex32>.asPtr(offset: Int = 0): Ptr<Complex32> = Complex32ArrayPtr(this, offset)

public fun Array<Complex32>.asMutPtr(offset: Int = 0): Ptr<Complex32> = Complex32ArrayPtr(this, offset)

public fun Array<Complex64>.asPtr(offset: Int = 0): Ptr<Complex64> = Complex64ArrayPtr(this, offset)

public fun Array<Complex64>.asMutPtr(offset: Int = 0): Ptr<Complex64> = Complex64ArrayPtr(this, offset)

public fun <T> MutableList<T>.asPtr(offset: Int = 0): Ptr<T> = ListPtr(this, offset)

public fun <T> MutableList<T>.asMutPtr(offset: Int = 0): Ptr<T> = ListPtr(this, offset)

public fun FloatArray.asPtr(offset: Int = 0): Ptr<Float> = FloatArrayPtr(this, offset)

public fun FloatArray.asMutPtr(offset: Int = 0): Ptr<Float> = FloatArrayPtr(this, offset)

public fun DoubleArray.asPtr(offset: Int = 0): Ptr<Double> = DoubleArrayPtr(this, offset)

public fun DoubleArray.asMutPtr(offset: Int = 0): Ptr<Double> = DoubleArrayPtr(this, offset)
