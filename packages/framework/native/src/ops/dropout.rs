use smallvec::smallvec;
use rand::Rng;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore};

pub fn dropout_forward(
    x: TensorId, rate: f32, training: bool,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    if !training || rate == 0.0 {
        return x;
    }

    let x_data = store.to_host(x);
    let shape = store.shape(x).to_vec();
    let size = x_data.len();
    let scale = 1.0 / (1.0 - rate);

    let mut rng = rand::thread_rng();
    let mut mask = vec![0.0f32; size];
    let mut out = vec![0.0f32; size];

    for i in 0..size {
        let keep = rng.gen::<f32>() >= rate;
        mask[i] = if keep { 1.0 } else { 0.0 };
        out[i] = x_data[i] * mask[i] * scale;
    }

    let mask_id = store.from_vec(mask, &shape);
    let out_id = store.from_vec(out, &shape);

    tape.record(TapeEntry {
        op: BackwardOp::Dropout, output_id: out_id,
        input_ids: smallvec![x],
        saved: SavedContext::DropoutMask(mask_id, rate),
    });
    out_id
}

pub fn dropout_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::DropoutMask(mask_id, rate) = saved {
        let grad_data = store.to_host(grad);
        let mask_data = store.to_host(*mask_id);
        let shape = store.shape(grad).to_vec();
        let scale = 1.0 / (1.0 - rate);

        let out: Vec<f32> = grad_data.iter().zip(&mask_data)
            .map(|(g, m)| g * m * scale).collect();
        vec![Some(store.from_vec(out, &shape))]
    } else { vec![None] }
}
