use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore};

fn gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu_grad_scalar(x: f32) -> f32 {
    let s = (2.0f32 / std::f32::consts::PI).sqrt();
    let inner = s * (x + 0.044715 * x * x * x);
    let tanh_inner = inner.tanh();
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let d_inner = s * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
}

pub fn gelu_forward(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let data: Vec<f32> = store.to_host(a).iter().map(|&x| gelu_scalar(x)).collect();
    let shape = store.shape(a).to_vec();
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Gelu, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(a),
    });
    out
}

pub fn gelu_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(inp) = saved {
        let inp_data = store.to_host(*inp);
        let grad_data = store.to_host(grad);
        let data: Vec<f32> = grad_data.iter().zip(&inp_data)
            .map(|(g, &x)| g * gelu_grad_scalar(x)).collect();
        let shape = store.shape(grad).to_vec();
        vec![Some(store.from_vec(data, &shape))]
    } else { vec![None] }
}

pub fn relu_forward(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let data: Vec<f32> = store.to_host(a).iter().map(|&x| x.max(0.0)).collect();
    let shape = store.shape(a).to_vec();
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Relu, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(a),
    });
    out
}

pub fn relu_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(inp) = saved {
        let inp_data = store.to_host(*inp);
        let grad_data = store.to_host(grad);
        let data: Vec<f32> = grad_data.iter().zip(&inp_data)
            .map(|(g, &x)| if x > 0.0 { *g } else { 0.0 }).collect();
        let shape = store.shape(grad).to_vec();
        vec![Some(store.from_vec(data, &shape))]
    } else { vec![None] }
}
