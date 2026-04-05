/**
 * CUDA matrix multiplication via cuBLAS FFI using koffi.
 *
 * Dynamically loads libcudart and libcublas at runtime — fails gracefully
 * if CUDA is not available (no GPU, no toolkit, etc.).
 *
 * All data is converted Float64 → Float32 for cuBLAS sgemm and back.
 */

import type { Storage } from './tensor_data.js';
import { createSharedStorage, shapeProduct } from './tensor_data.js';

/* ------------------------------------------------------------------ */
/*  State                                                              */
/* ------------------------------------------------------------------ */

let _initialized = false;
let _initFailed  = false;

let _cudaMalloc:  (out: [unknown], bytes: number) => number;
let _cudaFree:    (ptr: unknown) => number;
let _cudaMemcpy:  (dst: unknown, src: unknown, bytes: number, kind: number) => number;
let _cublasSgemm: (
    handle: unknown,
    transa: number, transb: number,
    m: number, n: number, k: number,
    alpha: Float32Array, A: unknown, lda: number,
    B: unknown, ldb: number,
    beta: Float32Array, C: unknown, ldc: number,
) => number;
let _cublasHandle: unknown;

const CUBLAS_OP_N = 0;
const H2D = 1;
const D2H = 2;

/* ------------------------------------------------------------------ */
/*  Init                                                               */
/* ------------------------------------------------------------------ */

interface KoffiLib {
    func(sig: string): (...args: unknown[]) => unknown;
}

function tryLoad(koffi: { load: (name: string) => KoffiLib }, ...names: string[]): KoffiLib {
    for (const name of names) {
        try { return koffi.load(name); } catch { /* try next */ }
    }
    throw new Error(`Could not load any of: ${names.join(', ')}`);
}

async function initCuda(): Promise<boolean> {
    if (_initFailed)  return false;
    if (_initialized) return true;

    try {
        const koffi = (await import('koffi')).default;

        const cudart = tryLoad(koffi, 'libcudart.so', 'libcudart.so.12');
        const cublas = tryLoad(koffi, 'libcublas.so', 'libcublas.so.12');

        _cudaMalloc = cudart.func(
            'int cudaMalloc(_Out_ void **devPtr, size_t size)',
        ) as typeof _cudaMalloc;
        _cudaFree = cudart.func('int cudaFree(void *devPtr)') as typeof _cudaFree;
        _cudaMemcpy = cudart.func(
            'int cudaMemcpy(void *dst, const void *src, size_t count, int kind)',
        ) as typeof _cudaMemcpy;

        const cublasCreate = cublas.func(
            'int cublasCreate_v2(_Out_ void **handle)',
        ) as (out: [unknown]) => number;
        _cublasSgemm = cublas.func(
            'int cublasSgemm_v2(void *handle, int transa, int transb, ' +
            'int m, int n, int k, ' +
            'const float *alpha, void *A, int lda, ' +
            'void *B, int ldb, ' +
            'const float *beta, void *C, int ldc)',
        ) as typeof _cublasSgemm;

        const handleOut: [unknown] = [null];
        const status = cublasCreate(handleOut);
        if (status !== 0) {
            console.warn(`cuBLAS create failed (status=${status})`);
            _initFailed = true;
            return false;
        }
        _cublasHandle = handleOut[0];
        _initialized = true;
        console.log('CUDA/cuBLAS initialised — GPU matmul enabled');
        return true;
    } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        console.warn(`CUDA init skipped: ${msg}`);
        _initFailed = true;
        return false;
    }
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

const _alpha = new Float32Array([1.0]);
const _beta  = new Float32Array([0.0]);

/**
 * Row-major C = A·B  (A is MxK, B is KxN, C is MxN).
 * cuBLAS is column-major, so we compute  C^T = B^T · A^T
 * which means: cublasSgemm(N, M, K, B, N, A, K, C, N).
 *
 * dA / dB / dC are pre-allocated device pointers.
 */
function sgemm(
    M: number, K: number, N: number,
    aHost: Float32Array, bHost: Float32Array,
    dA: unknown, dB: unknown, dC: unknown,
    cHost: Float32Array,
): void {
    _cudaMemcpy(dA, aHost, aHost.byteLength, H2D);
    _cudaMemcpy(dB, bHost, bHost.byteLength, H2D);

    _cublasSgemm(
        _cublasHandle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        _alpha, dB, N,
        dA, K,
        _beta, dC, N,
    );

    _cudaMemcpy(cHost, dC, cHost.byteLength, D2H);
}

/**
 * Map a flat output-batch ordinal to the corresponding input-batch
 * ordinal, respecting broadcasting rules.
 */
function batchIndex(
    outBatch: number,
    outBatchDims: number[],
    inputBatchDims: number[],
): number {
    const numDims  = outBatchDims.length;
    const inputLen = inputBatchDims.length;
    const offset   = numDims - inputLen;

    let remaining = outBatch;
    const outIdx = new Array<number>(numDims);
    for (let d = numDims - 1; d >= 0; d--) {
        outIdx[d] = remaining % outBatchDims[d]!;
        remaining = Math.floor(remaining / outBatchDims[d]!);
    }

    let inputOrdinal = 0;
    let inputStride  = 1;
    for (let d = inputLen - 1; d >= 0; d--) {
        const idx = inputBatchDims[d] === 1 ? 0 : outIdx[d + offset]!;
        inputOrdinal += idx * inputStride;
        inputStride  *= inputBatchDims[d]!;
    }
    return inputOrdinal;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/**
 * GPU matrix multiply via cuBLAS.
 *
 * Same interface as `cpuMatMul` in tensor_ops.ts.
 * Returns `null` when CUDA is not available so the caller can fall
 * back to the CPU path.
 */
export async function cudaMatMul(
    aStorage: Storage,  aBatchDims: number[], M: number, K: number,
    bStorage: Storage,  bBatchDims: number[], N: number,
    outBatchDims: number[],
): Promise<Storage | null> {
    const ok = await initCuda();
    if (!ok) return null;

    const outBatchSize = shapeProduct(outBatchDims);
    const outSize = outBatchSize * M * N;
    const outStorage = createSharedStorage(outSize);

    const aMK   = M * K;
    const bKN   = K * N;
    const outMN = M * N;

    const aF32 = new Float32Array(aStorage);
    const bF32 = new Float32Array(bStorage);

    const dA: [unknown] = [null];
    const dB: [unknown] = [null];
    const dC: [unknown] = [null];
    _cudaMalloc(dA, aMK * 4);
    _cudaMalloc(dB, bKN * 4);
    _cudaMalloc(dC, outMN * 4);

    try {
        const cBuf = new Float32Array(outMN);

        for (let batch = 0; batch < outBatchSize; batch++) {
            const aBatch = batchIndex(batch, outBatchDims, aBatchDims);
            const bBatch = batchIndex(batch, outBatchDims, bBatchDims);

            const aOff = aBatch * aMK;
            const bOff = bBatch * bKN;

            sgemm(
                M, K, N,
                aF32.subarray(aOff, aOff + aMK),
                bF32.subarray(bOff, bOff + bKN),
                dA[0], dB[0], dC[0],
                cBuf,
            );

            const outOff = batch * outMN;
            for (let i = 0; i < outMN; i++) {
                outStorage[outOff + i] = cBuf[i]!;
            }
        }
    } finally {
        _cudaFree(dA[0]);
        _cudaFree(dB[0]);
        _cudaFree(dC[0]);
    }

    return outStorage;
}

let _cublasDestroy: ((handle: unknown) => number) | null = null;

export async function destroyCuda(): Promise<void> {
    if (!_initialized || !_cublasHandle) return;
    try {
        if (!_cublasDestroy) {
            const koffi = (await import('koffi')).default;
            const cublas = tryLoad(koffi, 'libcublas.so', 'libcublas.so.12');
            _cublasDestroy = cublas.func('int cublasDestroy_v2(void *handle)') as
                (handle: unknown) => number;
        }
        _cublasDestroy(_cublasHandle);
        _cublasHandle = null;
        _initialized = false;
    } catch { /* ignore */ }
}
