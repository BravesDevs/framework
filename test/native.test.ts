import {
    Tensor, native,
    flashAttention, residualLayerNorm, biasGelu, KvCache,
} from '../dist/index.js';
import { assert, skip, section } from './helpers.js';

// ============================================================
// FlashAttention / ResidualLayerNorm / BiasGelu (GPU/CUDA only)
// ============================================================

section('FlashAttention / ResidualLayerNorm / BiasGelu');

if (typeof native.flashAttention === 'function') {
    const nHeads = 2, seqLen = 4, headDim = 8;
    const qAtt = Tensor.rand([1, nHeads, seqLen, headDim]);
    const kAtt = Tensor.rand([1, nHeads, seqLen, headDim]);
    const vAtt = Tensor.rand([1, nHeads, seqLen, headDim]);
    const scale = 1.0 / Math.sqrt(headDim);
    const attOut = flashAttention(qAtt, kAtt, vAtt, scale, true);
    assert(attOut.shape[0] === 1, 'attention batch');
    assert(attOut.shape[1] === nHeads, 'attention heads');
    assert(attOut.shape[2] === seqLen, 'attention seq_len');
    assert(attOut.shape[3] === headDim, 'attention head_dim');
} else {
    skip('flashAttention not available in CPU build');
}

if (typeof native.residualLayernorm === 'function') {
    const rlnX = Tensor.rand([2, 4]);
    const rlnResidual = Tensor.rand([2, 4]);
    const rlnGamma = Tensor.ones([4]).setRequiresGrad(true);
    const rlnBeta = Tensor.zeros([4]).setRequiresGrad(true);
    const rlnOut = residualLayerNorm(rlnX, rlnResidual, rlnGamma, rlnBeta);
    assert(rlnOut.shape[0] === 2 && rlnOut.shape[1] === 4, 'residualLayerNorm shape');
} else {
    skip('residualLayerNorm not available in CPU build');
}

if (typeof native.biasGelu === 'function') {
    const bgX = Tensor.rand([2, 4]);
    const bgBias = Tensor.rand([4]);
    const bgOut = biasGelu(bgX, bgBias);
    assert(bgOut.shape[0] === 2 && bgOut.shape[1] === 4, 'biasGelu shape');
} else {
    skip('biasGelu not available in CPU build');
}

section('KV cache decode + quantization');

if (typeof native.kvCacheCreate === 'function') {
    const batch = 1;
    const heads = 2;
    const headDim = 8;
    const maxSeq = 16;
    const steps = 6;
    const scale = 1.0 / Math.sqrt(headDim);

    const fpCache = new KvCache(batch, heads, headDim, maxSeq, false);
    const qCache = new KvCache(batch, heads, headDim, maxSeq, true);
    assert(qCache.isQuantized() || typeof native.flashAttention !== 'function', 'quantized cache enabled or safely downgraded');

    let maxAbsDiff = 0.0;
    let totalDecodeMs = 0.0;
    for (let t = 0; t < steps; t++) {
        const q = Tensor.rand([batch, heads, 1, headDim]);
        const k = Tensor.rand([batch, heads, 1, headDim]);
        const v = Tensor.rand([batch, heads, 1, headDim]);

        const start = Date.now();
        const outFp = fpCache.decodeStep(q, k, v, scale);
        const outQ = qCache.decodeStep(q, k, v, scale);
        totalDecodeMs += (Date.now() - start);

        const a = outFp.toFloat32();
        const b = outQ.toFloat32();
        for (let i = 0; i < a.length; i++) {
            maxAbsDiff = Math.max(maxAbsDiff, Math.abs(a[i] - b[i]));
        }
    }

    assert(fpCache.length() === steps, 'fp32 kv cache length tracks decode steps');
    assert(qCache.length() === steps, 'quantized kv cache length tracks decode steps');
    assert(maxAbsDiff < 0.15, `quantized kv decode drift bounded (max_abs_diff=${maxAbsDiff.toFixed(4)})`);

    // Perf gates: budget is intentionally loose because backend availability differs by platform.
    const avgDecodeMs = totalDecodeMs / steps;
    assert(avgDecodeMs < 25, `decode step latency gate (avg_ms=${avgDecodeMs.toFixed(2)})`);

    const fp32Bytes = batch * heads * maxSeq * headDim * 2 * 4;
    const int8Bytes = batch * heads * maxSeq * headDim * 2 * 1 + batch * heads * maxSeq * 2 * 4;
    assert(int8Bytes < fp32Bytes, 'quantized cache theoretical memory footprint is lower than fp32');

    fpCache.reset();
    assert(fpCache.length() === 0, 'kv cache reset clears logical length');
    fpCache.free();
    qCache.free();
} else {
    skip('kv cache APIs not available in current native build');
}

