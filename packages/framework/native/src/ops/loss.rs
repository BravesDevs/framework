use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore, shape_size};

/// Fused cross-entropy: softmax(logits) then -log(prob[target]), averaged.
/// logits: [B*T, V], targets: flat array of length B*T with class indices.
pub fn cross_entropy(
    logits: TensorId, targets: &[usize],
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let shape = store.shape(logits).to_vec();
    let ndim = shape.len();
    let v = shape[ndim - 1];
    let n = shape_size(&shape) / v;
    let data = store.to_host(logits);

    // Compute softmax and NLL in one pass
    let mut softmax_buf = vec![0.0f32; n * v];
    let mut loss_sum = 0.0f32;

    for i in 0..n {
        let off = i * v;
        let mut max_val = f32::NEG_INFINITY;
        for j in 0..v {
            max_val = max_val.max(data[off + j]);
        }
        let mut sum = 0.0f32;
        for j in 0..v {
            let e = (data[off + j] - max_val).exp();
            softmax_buf[off + j] = e;
            sum += e;
        }
        for j in 0..v {
            softmax_buf[off + j] /= sum;
        }
        let target = targets[i];
        loss_sum += -softmax_buf[off + target].ln();
    }

    let loss = loss_sum / n as f32;
    let loss_id = store.from_vec(vec![loss], &[1]);

    // Save softmax output for backward
    let sm_id = store.from_vec(softmax_buf, &shape);

    tape.record(TapeEntry {
        op: BackwardOp::CrossEntropy, output_id: loss_id,
        input_ids: smallvec![logits],
        saved: SavedContext::Indices(targets.to_vec(), n, v, sm_id),
    });
    loss_id
}

pub fn cross_entropy_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Indices(targets, n, v, sm_id) = saved {
        let grad_val = store.get_scalar(grad);
        let sm_data = store.to_host(*sm_id);
        let shape = store.shape(*sm_id).to_vec();

        let mut dlogits = sm_data.clone();
        for i in 0..*n {
            let off = i * v;
            dlogits[off + targets[i]] -= 1.0;
            let scale = grad_val / *n as f32;
            for j in 0..*v {
                dlogits[off + j] *= scale;
            }
        }
        vec![Some(store.from_vec(dlogits, &shape))]
    } else { vec![None] }
}
