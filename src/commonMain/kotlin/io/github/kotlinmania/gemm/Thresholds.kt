package io.github.kotlinmania.gemm

import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.ExperimentalAtomicApi

public const val DEFAULT_THREADING_THRESHOLD: Int = 48 * 48 * 256
public const val DEFAULT_RHS_PACKING_THRESHOLD: Int = 128
public const val DEFAULT_LHS_PACKING_THRESHOLD_SINGLE_THREAD: Int = 8
public const val DEFAULT_LHS_PACKING_THRESHOLD_MULTI_THREAD: Int = 16

@OptIn(ExperimentalAtomicApi::class)
private val threadingThreshold: AtomicInt = AtomicInt(DEFAULT_THREADING_THRESHOLD)

@OptIn(ExperimentalAtomicApi::class)
private val rhsPackingThreshold: AtomicInt = AtomicInt(DEFAULT_RHS_PACKING_THRESHOLD)

@OptIn(ExperimentalAtomicApi::class)
private val lhsPackingThresholdSingleThread: AtomicInt = AtomicInt(DEFAULT_LHS_PACKING_THRESHOLD_SINGLE_THREAD)

@OptIn(ExperimentalAtomicApi::class)
private val lhsPackingThresholdMultiThread: AtomicInt = AtomicInt(DEFAULT_LHS_PACKING_THRESHOLD_MULTI_THREAD)

@OptIn(ExperimentalAtomicApi::class)
public fun getThreadingThreshold(): Int = threadingThreshold.load()

@OptIn(ExperimentalAtomicApi::class)
public fun setThreadingThreshold(value: Int) {
    threadingThreshold.store(value)
}

@OptIn(ExperimentalAtomicApi::class)
public fun getRhsPackingThreshold(): Int = rhsPackingThreshold.load()

@OptIn(ExperimentalAtomicApi::class)
public fun setRhsPackingThreshold(value: Int) {
    rhsPackingThreshold.store(minOf(value, 256))
}

@OptIn(ExperimentalAtomicApi::class)
public fun getLhsPackingThresholdSingleThread(): Int = lhsPackingThresholdSingleThread.load()

@OptIn(ExperimentalAtomicApi::class)
public fun setLhsPackingThresholdSingleThread(value: Int) {
    lhsPackingThresholdSingleThread.store(minOf(value, 256))
}

@OptIn(ExperimentalAtomicApi::class)
public fun getLhsPackingThresholdMultiThread(): Int = lhsPackingThresholdMultiThread.load()

@OptIn(ExperimentalAtomicApi::class)
public fun setLhsPackingThresholdMultiThread(value: Int) {
    lhsPackingThresholdMultiThread.store(minOf(value, 256))
}
