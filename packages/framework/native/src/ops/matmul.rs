use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore, shape_size};

/// CPU matmul supporting batched dimensions.
/// A: [..., M, K], B: [..., K, N] → C: [..., M, N]
pub fn matmul(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let a_id = store.ensure_contiguous(a);
    let b_id = store.ensure_contiguous(b);
    let a_shape = store.shape(a_id).to_vec();
    let b_shape = store.shape(b_id).to_vec();

    assert!(a_shape.len() >= 2 && b_shape.len() >= 2, "matmul requires at least 2D tensors");
    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let k2 = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];
    assert_eq!(k, k2, "matmul inner dimensions must match: {} vs {}", k, k2);

    let a_batch: Vec<usize> = a_shape[..a_shape.len()-2].to_vec();
    let b_batch: Vec<usize> = b_shape[..b_shape.len()-2].to_vec();

    let out_batch = crate::utils::broadcast_shape(&a_batch, &b_batch);
    let batch_size = shape_size(&out_batch);

    let mut out_shape = out_batch.clone();
    out_shape.push(m);
    out_shape.push(n);

    let a_batch_size = shape_size(&a_batch);
    let b_batch_size = shape_size(&b_batch);

    let a_data = store.to_host(a_id);
    let b_data = store.to_host(b_id);
    let out_size = shape_size(&out_shape);
    let mut out = vec![0.0f32; out_size];

    let a_mat = m * k;
    let b_mat = k * n;
    let c_mat = m * n;

    for batch in 0..batch_size {
        let a_batch_idx = if a_batch_size == 1 { 0 } else { batch % a_batch_size };
        let b_batch_idx = if b_batch_size == 1 { 0 } else { batch % b_batch_size };

        let a_off = a_batch_idx * a_mat;
        let b_off = b_batch_idx * b_mat;
        let c_off = batch * c_mat;

        // ikj loop order for cache efficiency
        for i in 0..m {
            for kk in 0..k {
                let a_val = a_data[a_off + i * k + kk];
                for j in 0..n {
                    out[c_off + i * n + j] += a_val * b_data[b_off + kk * n + j];
                }
            }
        }
    }

    let out_id = store.from_vec(out, &out_shape);
    tape.record(TapeEntry {
        op: BackwardOp::MatMul, output_id: out_id, input_ids: smallvec![a, b],
        saved: SavedContext::Tensors(smallvec![a, b]),
    });
    out_id
}

pub fn matmul_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensors(ids) = saved {
        let a = ids[0]; let b = ids[1];
        let a_shape = store.shape(a).to_vec();
        let b_shape = store.shape(b).to_vec();

        // grad_a = grad @ B^T
        let b_t = transpose_last2(b, store);
        let grad_a = matmul_no_grad(grad, b_t, store);
        let grad_a_data = store.to_host(grad_a);
        let grad_a_shape = store.shape(grad_a).to_vec();
        let ga = crate::utils::unbroadcast(&grad_a_data, &grad_a_shape, &a_shape);
        let ga_id = store.from_vec(ga, &a_shape);

        // grad_b = A^T @ grad
        let a_t = transpose_last2(a, store);
        let grad_b = matmul_no_grad(a_t, grad, store);
        let grad_b_data = store.to_host(grad_b);
        let grad_b_shape = store.shape(grad_b).to_vec();
        let gb = crate::utils::unbroadcast(&grad_b_data, &grad_b_shape, &b_shape);
        let gb_id = store.from_vec(gb, &b_shape);

        vec![Some(ga_id), Some(gb_id)]
    } else { vec![None, None] }
}

fn transpose_last2(a: TensorId, store: &mut TensorStore) -> TensorId {
    let shape = store.shape(a).to_vec();
    let ndim = shape.len();
    let a_id = store.ensure_contiguous(a);
    let data = store.to_host(a_id);

    let m = shape[ndim - 2];
    let n = shape[ndim - 1];
    let batch_size: usize = shape[..ndim-2].iter().product::<usize>().max(1);

    let mut out = vec![0.0f32; data.len()];
    let mat_size = m * n;
    for b in 0..batch_size {
        let off = b * mat_size;
        for i in 0..m {
            for j in 0..n {
                out[off + j * m + i] = data[off + i * n + j];
            }
        }
    }

    let mut out_shape = shape[..ndim-2].to_vec();
    out_shape.push(n);
    out_shape.push(m);
    store.from_vec(out, &out_shape)
}

/// Matmul without recording to tape (used in backward).
fn matmul_no_grad(a: TensorId, b: TensorId, store: &mut TensorStore) -> TensorId {
    let mut dummy_tape = crate::autograd::Tape::new();
    dummy_tape.set_enabled(false);
    matmul(a, b, store, &mut dummy_tape)
}
