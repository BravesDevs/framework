use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore, compute_strides, shape_size};
use crate::utils::{broadcast_shape, unbroadcast, to_coord};

fn broadcast_binary(
    a: TensorId, b: TensorId, store: &mut TensorStore,
    f: fn(f32, f32) -> f32,
) -> (Vec<f32>, Vec<usize>) {
    let a_shape = store.shape(a).to_vec();
    let b_shape = store.shape(b).to_vec();
    let out_shape = broadcast_shape(&a_shape, &b_shape);
    let out_size = shape_size(&out_shape);
    let out_strides = compute_strides(&out_shape);

    let a_data = store.to_host(a);
    let b_data = store.to_host(b);
    let a_strides = compute_strides(&a_shape);
    let b_strides = compute_strides(&b_shape);

    let mut out = vec![0.0f32; out_size];
    let ndim = out_shape.len();
    let a_off = ndim - a_shape.len();
    let b_off = ndim - b_shape.len();

    for i in 0..out_size {
        let coord = to_coord(i, &out_shape, &out_strides);
        let mut ai = 0;
        for d in 0..a_shape.len() {
            let c = if a_shape[d] == 1 { 0 } else { coord[d + a_off] };
            ai += c * a_strides[d];
        }
        let mut bi = 0;
        for d in 0..b_shape.len() {
            let c = if b_shape[d] == 1 { 0 } else { coord[d + b_off] };
            bi += c * b_strides[d];
        }
        out[i] = f(a_data[ai], b_data[bi]);
    }
    (out, out_shape)
}

pub fn add(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let (data, shape) = broadcast_binary(a, b, store, |x, y| x + y);
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Add,
        output_id: out,
        input_ids: smallvec![a, b],
        saved: SavedContext::TensorsAndShape(smallvec![a, b], shape),
    });
    out
}

pub fn add_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorsAndShape(ids, _) = saved {
        let a_shape = store.shape(ids[0]).to_vec();
        let b_shape = store.shape(ids[1]).to_vec();
        let grad_data = store.to_host(grad);
        let grad_shape = store.shape(grad).to_vec();

        let ga = unbroadcast(&grad_data, &grad_shape, &a_shape);
        let gb = unbroadcast(&grad_data, &grad_shape, &b_shape);
        vec![Some(store.from_vec(ga, &a_shape)), Some(store.from_vec(gb, &b_shape))]
    } else { vec![None, None] }
}

pub fn mul(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let (data, shape) = broadcast_binary(a, b, store, |x, y| x * y);
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Mul,
        output_id: out,
        input_ids: smallvec![a, b],
        saved: SavedContext::Tensors(smallvec![a, b]),
    });
    out
}

pub fn mul_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensors(ids) = saved {
        let a = ids[0]; let b = ids[1];
        let a_shape = store.shape(a).to_vec();
        let b_shape = store.shape(b).to_vec();
        let grad_shape = store.shape(grad).to_vec();
        let a_data = store.to_host(a);
        let b_data = store.to_host(b);
        let grad_data = store.to_host(grad);
        let grad_size = shape_size(&grad_shape);

        let out_strides = compute_strides(&grad_shape);
        let ndim = grad_shape.len();

        let mut ga_full = vec![0.0f32; grad_size];
        let mut gb_full = vec![0.0f32; grad_size];
        let a_strides_o = compute_strides(&a_shape);
        let b_strides_o = compute_strides(&b_shape);
        let a_off = ndim - a_shape.len();
        let b_off = ndim - b_shape.len();

        for i in 0..grad_size {
            let coord = to_coord(i, &grad_shape, &out_strides);
            let mut ai = 0;
            for d in 0..a_shape.len() {
                let c = if a_shape[d] == 1 { 0 } else { coord[d + a_off] };
                ai += c * a_strides_o[d];
            }
            let mut bi = 0;
            for d in 0..b_shape.len() {
                let c = if b_shape[d] == 1 { 0 } else { coord[d + b_off] };
                bi += c * b_strides_o[d];
            }
            ga_full[i] = grad_data[i] * b_data[bi];
            gb_full[i] = grad_data[i] * a_data[ai];
        }

        let ga = unbroadcast(&ga_full, &grad_shape, &a_shape);
        let gb = unbroadcast(&gb_full, &grad_shape, &b_shape);
        vec![Some(store.from_vec(ga, &a_shape)), Some(store.from_vec(gb, &b_shape))]
    } else { vec![None, None] }
}

pub fn sub(a: TensorId, b: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let (data, shape) = broadcast_binary(a, b, store, |x, y| x - y);
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Sub,
        output_id: out,
        input_ids: smallvec![a, b],
        saved: SavedContext::TensorsAndShape(smallvec![a, b], shape),
    });
    out
}

pub fn sub_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorsAndShape(ids, _) = saved {
        let a_shape = store.shape(ids[0]).to_vec();
        let b_shape = store.shape(ids[1]).to_vec();
        let grad_data = store.to_host(grad);
        let grad_shape = store.shape(grad).to_vec();

        let ga = unbroadcast(&grad_data, &grad_shape, &a_shape);
        let neg_grad: Vec<f32> = grad_data.iter().map(|x| -x).collect();
        let gb = unbroadcast(&neg_grad, &grad_shape, &b_shape);
        vec![Some(store.from_vec(ga, &a_shape)), Some(store.from_vec(gb, &b_shape))]
    } else { vec![None, None] }
}

pub fn neg(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let data: Vec<f32> = store.to_host(a).iter().map(|x| -x).collect();
    let shape = store.shape(a).to_vec();
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Neg, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::None,
    });
    out
}

pub fn neg_backward(grad: TensorId, _saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    let data: Vec<f32> = store.to_host(grad).iter().map(|x| -x).collect();
    let shape = store.shape(grad).to_vec();
    vec![Some(store.from_vec(data, &shape))]
}

pub fn mul_scalar(a: TensorId, s: f32, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let data: Vec<f32> = store.to_host(a).iter().map(|x| x * s).collect();
    let shape = store.shape(a).to_vec();
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::MulScalar, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::TensorAndScalar(a, s),
    });
    out
}

pub fn mul_scalar_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::TensorAndScalar(_, s) = saved {
        let data: Vec<f32> = store.to_host(grad).iter().map(|x| x * s).collect();
        let shape = store.shape(grad).to_vec();
        vec![Some(store.from_vec(data, &shape))]
    } else { vec![None] }
}

pub fn exp(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let data: Vec<f32> = store.to_host(a).iter().map(|x| x.exp()).collect();
    let shape = store.shape(a).to_vec();
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Exp, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(out),
    });
    out
}

pub fn exp_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(out) = saved {
        let out_data = store.to_host(*out);
        let grad_data = store.to_host(grad);
        let data: Vec<f32> = grad_data.iter().zip(&out_data).map(|(g, o)| g * o).collect();
        let shape = store.shape(grad).to_vec();
        vec![Some(store.from_vec(data, &shape))]
    } else { vec![None] }
}

pub fn log(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let data: Vec<f32> = store.to_host(a).iter().map(|x| x.ln()).collect();
    let shape = store.shape(a).to_vec();
    let out = store.from_vec(data, &shape);
    tape.record(TapeEntry {
        op: BackwardOp::Log, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Tensor(a),
    });
    out
}

pub fn log_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Tensor(inp) = saved {
        let inp_data = store.to_host(*inp);
        let grad_data = store.to_host(grad);
        let data: Vec<f32> = grad_data.iter().zip(&inp_data).map(|(g, x)| g / x).collect();
        let shape = store.shape(grad).to_vec();
        vec![Some(store.from_vec(data, &shape))]
    } else { vec![None] }
}
