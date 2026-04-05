/**
 * CUDA matrix multiplication via cuBLAS FFI using koffi.
 *
 * Dynamically loads libcudart and libcublas at runtime — fails gracefully
 * if CUDA is not available (no GPU, no toolkit, etc.).
 *
 * Features:
 *  - Device buffer pooling (reuses cudaMalloc buffers across calls)
 *  - cublasSgemmStridedBatched for non-broadcast batched matmuls
 *  - Float64 → Float32 conversion for cuBLAS, Float32 → Float64 back
 */

import type { Storage } from './tensor_data.js';
import { createSharedStorage, shapeProduct } from './tensor_data.js';

/* ------------------------------------------------------------------ */
/*  Types & State                                                      */
/* ------------------------------------------------------------------ */

interface KoffiLib {
    func(sig: string): (...args: unknown[]) => unknown;
}

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
let _cublasSgemmStridedBatched: (
    handle: unknown,
    transa: number, transb: number,
    m: number, n: number, k: number,
    alpha: Float32Array, A: unknown, lda: number, strideA: number,
    B: unknown, ldb: number, strideB: number,
    beta: Float32Array, C: unknown, ldc: number, strideC: number,
    batchCount: number,
) => number;
let _cublasHandle: unknown;
let _cublasDestroy: ((handle: unknown) => number) | null = null;

const CUBLAS_OP_N = 0;
const H2D = 1;
const D2H = 2;

/* ------------------------------------------------------------------ */
/*  Device Buffer Pool                                                 */
/* ------------------------------------------------------------------ */

const _bufferPool = new Map<number, unknown[]>();

function poolAlloc(bytes: number): unknown {
    const bucket = _bufferPool.get(bytes);
    if (bucket && bucket.length > 0) {
        return bucket.pop()!;
    }
    const out: [unknown] = [null];
    _cudaMalloc(out, bytes);
    return out[0];
}

function poolRelease(ptr: unknown, bytes: number): void {
    let bucket = _bufferPool.get(bytes);
    if (!bucket) {
        bucket = [];
        _bufferPool.set(bytes, bucket);
    }
    bucket.push(ptr);
}

function poolFreeAll(): void {
    for (const [, bucket] of _bufferPool) {
        for (const ptr of bucket) _cudaFree(ptr);
    }
    _bufferPool.clear();
}

/* ------------------------------------------------------------------ */
/*  Init                                                               */
/* ------------------------------------------------------------------ */

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
        _cublasSgemmStridedBatched = cublas.func(
            'int cublasSgemmStridedBatched(void *handle, int transa, int transb, ' +
            'int m, int n, int k, ' +
            'const float *alpha, void *A, int lda, long long strideA, ' +
            'void *B, int ldb, long long strideB, ' +
            'const float *beta, void *C, int ldc, long long strideC, ' +
            'int batchCount)',
        ) as typeof _cublasSgemmStridedBatched;
        _cublasDestroy = cublas.func(
            'int cublasDestroy_v2(void *handle)',
        ) as (handle: unknown) => number;

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

function dimsEqual(a: number[], b: number[]): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

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

    const canStridedBatch = outBatchSize > 1 && dimsEqual(aBatchDims, bBatchDims)
                            && dimsEqual(aBatchDims, outBatchDims);

    if (canStridedBatch) {
        // Upload all batch data at once, one strided batched GEMM call
        const totalABytes = outBatchSize * aMK * 4;
        const totalBBytes = outBatchSize * bKN * 4;
        const totalCBytes = outBatchSize * outMN * 4;

        const dA = poolAlloc(totalABytes);
        const dB = poolAlloc(totalBBytes);
        const dC = poolAlloc(totalCBytes);

        _cudaMemcpy(dA, aF32, totalABytes, H2D);
        _cudaMemcpy(dB, bF32, totalBBytes, H2D);

        // Row-major trick: C^T = B^T · A^T
        _cublasSgemmStridedBatched(
            _cublasHandle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            _alpha, dB, N, bKN,
            dA, K, aMK,
            _beta, dC, N, outMN,
            outBatchSize,
        );

        const cF32 = new Float32Array(outSize);
        _cudaMemcpy(cF32, dC, totalCBytes, D2H);

        for (let i = 0; i < outSize; i++) {
            outStorage[i] = cF32[i]!;
        }

        poolRelease(dA, totalABytes);
        poolRelease(dB, totalBBytes);
        poolRelease(dC, totalCBytes);
    } else {
        // Per-batch fallback (handles broadcasting)
        const aBytes = aMK * 4;
        const bBytes = bKN * 4;
        const cBytes = outMN * 4;

        const dA = poolAlloc(aBytes);
        const dB = poolAlloc(bBytes);
        const dC = poolAlloc(cBytes);

        const cBuf = new Float32Array(outMN);

        for (let batch = 0; batch < outBatchSize; batch++) {
            const aBatch = batchIndex(batch, outBatchDims, aBatchDims);
            const bBatch = batchIndex(batch, outBatchDims, bBatchDims);

            const aOff = aBatch * aMK;
            const bOff = bBatch * bKN;

            const aSub = aF32.subarray(aOff, aOff + aMK);
            const bSub = bF32.subarray(bOff, bOff + bKN);

            _cudaMemcpy(dA, aSub, aBytes, H2D);
            _cudaMemcpy(dB, bSub, bBytes, H2D);

            _cublasSgemm(
                _cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                _alpha, dB, N,
                dA, K,
                _beta, dC, N,
            );

            _cudaMemcpy(cBuf, dC, cBytes, D2H);

            const outOff = batch * outMN;
            for (let i = 0; i < outMN; i++) {
                outStorage[outOff + i] = cBuf[i]!;
            }
        }

        poolRelease(dA, aBytes);
        poolRelease(dB, bBytes);
        poolRelease(dC, cBytes);
    }

    return outStorage;
}

export async function destroyCuda(): Promise<void> {
    if (!_initialized || !_cublasHandle) return;
    try {
        poolFreeAll();
        if (_cublasDestroy) {
            _cublasDestroy(_cublasHandle);
        }
        _cublasHandle = null;
        _initialized = false;
    } catch { /* ignore */ }
}
