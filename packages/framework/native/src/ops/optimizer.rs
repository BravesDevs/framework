use crate::tensor::{TensorId, TensorStore};

pub fn adamw_step(
    param_ids: &[TensorId],
    lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32,
    step: u32, store: &mut TensorStore,
) {
    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);

    for &pid in param_ids {
        let grad_id = match store.get(pid).grad {
            Some(g) => g,
            None => continue,
        };

        let size = store.size(pid);
        let grad_data = store.to_host(grad_id);

        // Lazily initialize optimizer state
        if store.get(pid).adam_m.is_none() {
            store.get_mut(pid).adam_m = Some(vec![0.0f32; size]);
            store.get_mut(pid).adam_v = Some(vec![0.0f32; size]);
        }

        let param = store.get_mut(pid);
        let m = param.adam_m.as_mut().unwrap();
        let v = param.adam_v.as_mut().unwrap();
        let data = &mut param.data;

        for i in 0..size {
            let g = grad_data[i];

            // Weight decay (decoupled)
            if weight_decay > 0.0 {
                data[i] *= 1.0 - lr * weight_decay;
            }

            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;

            data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

pub fn grad_norm(param_ids: &[TensorId], store: &TensorStore) -> f32 {
    let mut norm_sq = 0.0f64;
    for &pid in param_ids {
        if let Some(grad_id) = store.get(pid).grad {
            let data = store.data(grad_id);
            for &v in data {
                norm_sq += (v as f64) * (v as f64);
            }
        }
    }
    (norm_sq as f32).sqrt()
}

pub fn clip_grad_norm(param_ids: &[TensorId], max_norm: f32, store: &mut TensorStore) {
    let norm = grad_norm(param_ids, store);
    if norm > max_norm {
        let scale = max_norm / norm;
        for &pid in param_ids {
            if let Some(grad_id) = store.get(pid).grad {
                let data = store.data_mut(grad_id);
                for v in data.iter_mut() {
                    *v *= scale;
                }
            }
        }
    }
}
