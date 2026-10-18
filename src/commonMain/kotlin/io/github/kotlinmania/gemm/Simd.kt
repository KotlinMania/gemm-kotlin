package io.github.kotlinmania.gemm

import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.ExperimentalAtomicApi

public const val DEFAULT_WASM_SIMD128: Boolean = false

@OptIn(ExperimentalAtomicApi::class)
private val wasmSimd128: AtomicBoolean = AtomicBoolean(DEFAULT_WASM_SIMD128)

@OptIn(ExperimentalAtomicApi::class)
public fun getWasmSimd128(): Boolean = wasmSimd128.load()

@OptIn(ExperimentalAtomicApi::class)
public fun setWasmSimd128(enable: Boolean) {
    wasmSimd128.store(enable)
}
