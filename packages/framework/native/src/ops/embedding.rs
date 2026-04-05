use smallvec::smallvec;
use crate::autograd::{BackwardOp, SavedContext, Tape, TapeEntry};
use crate::tensor::{TensorId, TensorStore};

pub fn embedding_forward(
    weight: TensorId, indices: &[usize], batch: usize, seq_len: usize,
    store: &mut TensorStore, tape: &mut Tape,
) -> TensorId {
    let w_shape = store.shape(weight).to_vec();
    let embed_dim = w_shape[1];
    let w_data = store.to_host(weight);

    let mut out = vec![0.0f32; batch * seq_len * embed_dim];
    for b in 0..batch {
        for t in 0..seq_len {
            let idx = indices[b * seq_len + t];
            let src_off = idx * embed_dim;
            let dst_off = (b * seq_len + t) * embed_dim;
            out[dst_off..dst_off + embed_dim].copy_from_slice(&w_data[src_off..src_off + embed_dim]);
        }
    }

    let out_shape = vec![batch, seq_len, embed_dim];
    let out_id = store.from_vec(out, &out_shape);

    tape.record(TapeEntry {
        op: BackwardOp::Embedding, output_id: out_id,
        input_ids: smallvec![weight],
        saved: SavedContext::Indices(indices.to_vec(), batch, seq_len, weight),
    });
    out_id
}

pub fn embedding_backward(grad: TensorId, saved: &SavedContext, store: &mut TensorStore) -> Vec<Option<TensorId>> {
    if let SavedContext::Indices(indices, batch, seq_len, weight_id) = saved {
        let w_shape = store.shape(*weight_id).to_vec();
        let vocab_size = w_shape[0];
        let embed_dim = w_shape[1];
        let grad_data = store.to_host(grad);

        let mut dw = vec![0.0f32; vocab_size * embed_dim];
        for b in 0..*batch {
            for t in 0..*seq_len {
                let idx = indices[b * seq_len + t];
                let src_off = (b * seq_len + t) * embed_dim;
                let dst_off = idx * embed_dim;
                for j in 0..embed_dim {
                    dw[dst_off + j] += grad_data[src_off + j];
                }
            }
        }

        vec![Some(store.from_vec(dw, &w_shape))]
    } else { vec![None] }
}
