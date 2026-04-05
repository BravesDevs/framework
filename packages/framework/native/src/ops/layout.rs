use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore, compute_strides, shape_size};

pub fn view(a: TensorId, new_shape: &[usize], store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let orig_shape = store.shape(a).to_vec();
    let a_id = store.ensure_contiguous(a);
    let data = store.to_host(a_id);
    let out = store.from_vec(data, new_shape);
    tape.record(TapeEntry {
        op: BackwardOp::View, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Shape(orig_shape),
    });
    out
}

pub fn view_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Shape(orig_shape) = saved {
        let data = store.to_host(grad);
        vec![Some(store.from_vec(data, orig_shape))]
    } else { vec![None] }
}

pub fn permute(a: TensorId, dims: &[usize], store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let a_shape = store.shape(a).to_vec();
    let a_strides = store.get(a).strides.clone();
    let ndim = a_shape.len();

    let mut new_shape = vec![0usize; ndim];
    let mut new_strides = vec![0usize; ndim];
    for i in 0..ndim {
        new_shape[i] = a_shape[dims[i]];
        new_strides[i] = a_strides[dims[i]];
    }

    let data = store.get(a).data.clone();
    let size = store.size(a);
    let out = store.insert_raw(data, new_shape.clone(), new_strides, size);

    tape.record(TapeEntry {
        op: BackwardOp::Permute, output_id: out, input_ids: smallvec![a],
        saved: SavedContext::Permutation(dims.to_vec(), a_shape),
    });
    out
}

pub fn permute_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Permutation(order, orig_shape) = saved {
        let ndim = order.len();
        let mut inv = vec![0usize; ndim];
        for i in 0..ndim {
            inv[order[i]] = i;
        }
        let grad_contig = store.ensure_contiguous(grad);
        let data = store.to_host(grad_contig);
        let grad_shape = store.shape(grad_contig).to_vec();

        let src_strides = compute_strides(&grad_shape);
        let size = shape_size(&grad_shape);

        let mut out = vec![0.0f32; size];
        let out_strides = compute_strides(orig_shape);

        for i in 0..size {
            let mut coord = vec![0usize; ndim];
            let mut rem = i;
            for d in 0..ndim {
                coord[d] = rem / src_strides[d];
                rem %= src_strides[d];
            }
            let mut out_idx = 0;
            for d in 0..ndim {
                out_idx += coord[inv[d]] * out_strides[d];
            }
            out[out_idx] = data[i];
        }
        vec![Some(store.from_vec(out, orig_shape))]
    } else { vec![None] }
}

pub fn contiguous(a: TensorId, store: &mut TensorStore, tape: &mut Tape) -> TensorId {
    let out = store.ensure_contiguous(a);
    if out != a {
        tape.record(TapeEntry {
            op: BackwardOp::Contiguous, output_id: out, input_ids: smallvec![a],
            saved: SavedContext::None,
        });
    }
    out
}

pub fn contiguous_backward(grad: TensorId, _saved: &SavedContext, _store: &mut TensorStore) -> Vec<Option<TensorId>> {
    vec![Some(grad)]
}

// Helper: allow inserting a tensor with custom strides (for permute)
impl TensorStore {
    pub fn insert_raw(&mut self, data: Vec<f32>, shape: Vec<usize>, strides: Vec<usize>, size: usize) -> TensorId {
        use crate::tensor::GpuTensor;
        let t = GpuTensor {
            data, shape, strides, size,
            requires_grad: false, grad: None, adam_m: None, adam_v: None,
        };
        if let Some(id) = self.free_ids.pop() {
            self.tensors[id] = Some(t);
            id
        } else {
            let id = self.tensors.len();
            self.tensors.push(Some(t));
            id
        }
    }
}
