use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore, shape_size};

pub fn softmax(a: TensorId, dim: i32, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let a_shape = store.shape(a).to_vec();
    let ndim = a_shape.len();
    let d = if dim < 0 { (ndim as i32 + dim) as usize } else { dim as usize };
    let a_data = store.to_host(a);
    let total = shape_size(&a_shape);
    let dim_size = a_shape[d];

    let outer: usize = a_shape[..d].iter().product::<usize>().max(1);
    let inner: usize = a_shape[d+1..].iter().product::<usize>().max(1);

    let mut out = vec![0.0f32; total];

    for o in 0..outer {
        for i in 0..inner {
            let mut max_val = f32::NEG_INFINITY;
            for s in 0..dim_size {
                let idx = o * dim_size * inner + s * inner + i;
                max_val = max_val.max(a_data[idx]);
            }
            let mut sum = 0.0f32;
            for s in 0..dim_size {
                let idx = o * dim_size * inner + s * inner + i;
                let e = (a_data[idx] - max_val).exp();
                out[idx] = e;
                sum += e;
            }
            for s in 0..dim_size {
                let idx = o * dim_size * inner + s * inner + i;
                out[idx] /= sum;
            }
        }
    }

    let out_id = store.from_vec(out, &a_shape);
    tape.record(TapeEntry {
        op: BackwardOp::Softmax, output_id: out_id, input_ids: smallvec![a],
        saved: SavedContext::ScalarAndTensor(dim as f32, out_id),
    });
    out_id
}

pub fn softmax_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::ScalarAndTensor(dim_f, sm_out) = saved {
        let d = *dim_f as usize;
        let sm_data = store.to_host(*sm_out);
        let grad_data = store.to_host(grad);
        let shape = store.shape(*sm_out).to_vec();
        let total = shape_size(&shape);
        let dim_size = shape[d];
        let outer: usize = shape[..d].iter().product::<usize>().max(1);
        let inner: usize = shape[d+1..].iter().product::<usize>().max(1);

        let mut out = vec![0.0f32; total];
        for o in 0..outer {
            for i in 0..inner {
                let mut dot = 0.0f32;
                for s in 0..dim_size {
                    let idx = o * dim_size * inner + s * inner + i;
                    dot += grad_data[idx] * sm_data[idx];
                }
                for s in 0..dim_size {
                    let idx = o * dim_size * inner + s * inner + i;
                    out[idx] = sm_data[idx] * (grad_data[idx] - dot);
                }
            }
        }
        vec![Some(store.from_vec(out, &shape))]
    } else { vec![None] }
}

pub fn layernorm(
    x: TensorId, gamma: TensorId, beta: TensorId, eps: f32,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let x_shape = store.shape(x).to_vec();
    let ndim = x_shape.len();
    let c = x_shape[ndim - 1];
    let n = shape_size(&x_shape) / c;

    let x_data = store.to_host(x);
    let gamma_data = store.to_host(gamma);
    let beta_data = store.to_host(beta);

    let mut mean_buf = vec![0.0f32; n];
    let mut rstd_buf = vec![0.0f32; n];
    let mut out = vec![0.0f32; n * c];

    for row in 0..n {
        let off = row * c;
        let mut sum = 0.0f32;
        for j in 0..c { sum += x_data[off + j]; }
        let m = sum / c as f32;
        mean_buf[row] = m;

        let mut var = 0.0f32;
        for j in 0..c {
            let d = x_data[off + j] - m;
            var += d * d;
        }
        let rstd = 1.0 / (var / c as f32 + eps).sqrt();
        rstd_buf[row] = rstd;

        for j in 0..c {
            out[off + j] = (x_data[off + j] - m) * rstd * gamma_data[j] + beta_data[j];
        }
    }

    let mean_id = store.from_vec(mean_buf, &[n]);
    let rstd_id = store.from_vec(rstd_buf, &[n]);
    let out_id = store.from_vec(out, &x_shape);

    tape.record(TapeEntry {
        op: BackwardOp::LayerNorm, output_id: out_id,
        input_ids: smallvec![x, gamma, beta],
        saved: SavedContext::Tensors(smallvec![x, gamma, mean_id, rstd_id]),
    });
    out_id
}

pub fn layernorm_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensors(ids) = saved {
        let x = ids[0]; let gamma = ids[1]; let mean_id = ids[2]; let rstd_id = ids[3];
        let x_shape = store.shape(x).to_vec();
        let ndim = x_shape.len();
        let c = x_shape[ndim - 1];
        let n = shape_size(&x_shape) / c;

        let x_data = store.to_host(x);
        let gamma_data = store.to_host(gamma);
        let mean_data = store.to_host(mean_id);
        let rstd_data = store.to_host(rstd_id);
        let grad_data = store.to_host(grad);

        let mut dx = vec![0.0f32; n * c];
        let mut dgamma = vec![0.0f32; c];
        let mut dbeta = vec![0.0f32; c];

        for row in 0..n {
            let off = row * c;
            let m = mean_data[row];
            let rstd = rstd_data[row];

            let mut sum_dg = 0.0f32;
            let mut sum_dgx = 0.0f32;
            for j in 0..c {
                let xhat = (x_data[off + j] - m) * rstd;
                let dg = grad_data[off + j] * gamma_data[j];
                sum_dg += dg;
                sum_dgx += dg * xhat;
                dgamma[j] += grad_data[off + j] * xhat;
                dbeta[j] += grad_data[off + j];
            }

            for j in 0..c {
                let xhat = (x_data[off + j] - m) * rstd;
                let dg = grad_data[off + j] * gamma_data[j];
                dx[off + j] = rstd * (dg - sum_dg / c as f32 - xhat * sum_dgx / c as f32);
            }
        }

        let gamma_shape = store.shape(gamma).to_vec();
        let _beta_shape = store.shape(ids[2]).to_vec();
        vec![
            Some(store.from_vec(dx, &x_shape)),
            Some(store.from_vec(dgamma, &gamma_shape)),
            Some(store.from_vec(dbeta, &gamma_shape)),
        ]
    } else { vec![None, None, None] }
}
