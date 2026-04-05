use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore, compute_strides, shape_size};

fn resolve_dim(dim: i32, ndim: usize) -> usize {
    if dim < 0 { (ndim as i32 + dim) as usize } else { dim as usize }
}

pub fn sum(a: TensorId, dim: i32, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let a_shape = store.shape(a).to_vec();
    let d = resolve_dim(dim, a_shape.len());
    let a_data = store.to_host(a);

    let mut out_shape = a_shape.clone();
    out_shape[d] = 1;
    let out_size = shape_size(&out_shape);
    let mut out = vec![0.0f32; out_size];

    let a_strides = compute_strides(&a_shape);
    let out_strides = compute_strides(&out_shape);
    let total = shape_size(&a_shape);

    for i in 0..total {
        let mut out_idx = 0;
        let mut rem = i;
        for dd in 0..a_shape.len() {
            let coord = rem / a_strides[dd];
            rem %= a_strides[dd];
            let c = if dd == d { 0 } else { coord };
            out_idx += c * out_strides[dd];
        }
        out[out_idx] += a_data[i];
    }

    let out_id = store.from_vec(out, &out_shape);
    tape.record(TapeEntry {
        op: BackwardOp::Sum, output_id: out_id, input_ids: smallvec![a],
        saved: SavedContext::TensorAndShape(a, a_shape),
    });
    out_id
}

pub fn sum_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorAndShape(_, orig_shape) = saved {
        let grad_data = store.to_host(grad);
        let grad_shape = store.shape(grad).to_vec();
        let orig_size = shape_size(orig_shape);
        let grad_strides = compute_strides(&grad_shape);
        let orig_strides = compute_strides(orig_shape);
        let ndim = orig_shape.len();

        let mut out = vec![0.0f32; orig_size];
        for i in 0..orig_size {
            let mut grad_idx = 0;
            let mut rem = i;
            for d in 0..ndim {
                let coord = rem / orig_strides[d];
                rem %= orig_strides[d];
                let c = if grad_shape[d] == 1 { 0 } else { coord };
                grad_idx += c * grad_strides[d];
            }
            out[i] = grad_data[grad_idx];
        }
        vec![Some(store.from_vec(out, orig_shape))]
    } else { vec![None] }
}

pub fn mean(a: TensorId, dim: i32, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let a_shape = store.shape(a).to_vec();
    let d = resolve_dim(dim, a_shape.len());
    let count = a_shape[d] as f32;

    let sum_id = sum(a, dim, store, tape);
    // Remove the sum tape entry and replace with mean
    let last = tape.is_enabled();
    if last {
        // Pop the sum entry, we'll replace it
    }
    let sum_data: Vec<f32> = store.to_host(sum_id).iter().map(|x| x / count).collect();
    let shape = store.shape(sum_id).to_vec();
    let out = store.from_vec(sum_data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Mean, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::TensorAndShape(a, a_shape),
    });
    out
}

pub fn mean_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorAndShape(_, orig_shape) = saved {
        let grad_data = store.to_host(grad);
        let grad_shape = store.shape(grad).to_vec();
        let orig_size = shape_size(orig_shape);
        let grad_strides = compute_strides(&grad_shape);
        let orig_strides = compute_strides(orig_shape);
        let ndim = orig_shape.len();

        // Find which dim was reduced
        let mut reduced_dim_size = 1.0f32;
        for d in 0..ndim {
            if grad_shape[d] == 1 && orig_shape[d] > 1 {
                reduced_dim_size = orig_shape[d] as f32;
            }
        }

        let mut out = vec![0.0f32; orig_size];
        for i in 0..orig_size {
            let mut grad_idx = 0;
            let mut rem = i;
            for d in 0..ndim {
                let coord = rem / orig_strides[d];
                rem %= orig_strides[d];
                let c = if grad_shape[d] == 1 { 0 } else { coord };
                grad_idx += c * grad_strides[d];
            }
            out[i] = grad_data[grad_idx] / reduced_dim_size;
        }
        vec![Some(store.from_vec(out, orig_shape))]
    } else { vec![None] }
}

pub fn max(a: TensorId, dim: i32, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let a_shape = store.shape(a).to_vec();
    let d = resolve_dim(dim, a_shape.len());
    let a_data = store.to_host(a);

    let mut out_shape = a_shape.clone();
    out_shape[d] = 1;
    let out_size = shape_size(&out_shape);
    let mut out = vec![f32::NEG_INFINITY; out_size];

    let a_strides = compute_strides(&a_shape);
    let out_strides = compute_strides(&out_shape);
    let total = shape_size(&a_shape);

    for i in 0..total {
        let mut out_idx = 0;
        let mut rem = i;
        for dd in 0..a_shape.len() {
            let coord = rem / a_strides[dd];
            rem %= a_strides[dd];
            let c = if dd == d { 0 } else { coord };
            out_idx += c * out_strides[dd];
        }
        if a_data[i] > out[out_idx] {
            out[out_idx] = a_data[i];
        }
    }

    let out_id = store.from_vec(out, &out_shape);
    tape.record(TapeEntry {
        op: BackwardOp::Max, output_id: out_id, input_ids: smallvec![a],
        saved: SavedContext::Tensors(smallvec![a, out_id]),
    });
    out_id
}

pub fn max_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensors(ids) = saved {
        let inp = ids[0]; let max_out = ids[1];
        let inp_data = store.to_host(inp);
        let max_data = store.to_host(max_out);
        let inp_shape = store.shape(inp).to_vec();
        let max_shape = store.shape(max_out).to_vec();
        let grad_data = store.to_host(grad);

        let inp_size = shape_size(&inp_shape);
        let inp_strides = compute_strides(&inp_shape);
        let max_strides = compute_strides(&max_shape);
        let ndim = inp_shape.len();

        let mut out = vec![0.0f32; inp_size];
        for i in 0..inp_size {
            let mut max_idx = 0;
            let mut rem = i;
            for d in 0..ndim {
                let coord = rem / inp_strides[d];
                rem %= inp_strides[d];
                let c = if max_shape[d] == 1 { 0 } else { coord };
                max_idx += c * max_strides[d];
            }
            if (inp_data[i] - max_data[max_idx]).abs() < 1e-6 {
                out[i] = grad_data[max_idx];
            }
        }
        vec![Some(store.from_vec(out, &inp_shape))]
    } else { vec![None] }
}
