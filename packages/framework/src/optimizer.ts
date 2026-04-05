import { Tensor, native } from "./tensor.js";
import { Parameter } from "./module.js";

export type ParameterValue = Tensor;

export class Optimizer {
    parameters: Parameter<Tensor>[];

    constructor(parameters: Parameter<Tensor>[]) {
        this.parameters = parameters;
    }
}

export class SGD extends Optimizer {
    lr: number;

    constructor(parameters: Parameter<Tensor>[], lr: number = 1.0) {
        super(parameters);
        this.lr = lr;
    }

    zeroGrad() {
        const ids = this.parameters.map(p => p.value._id);
        native.zeroGrad(ids);
    }

    step() {
        // Simple SGD: p = p - lr * grad
        for (const p of this.parameters) {
            const grad = p.value.grad;
            if (!grad) continue;
            const updated = p.value.sub(grad.mul(this.lr));
            p.update(updated as any);
        }
    }
}

export class Adam extends Optimizer {
    lr: number;
    beta1: number;
    beta2: number;
    eps: number;
    weightDecay: number;
    t: number = 0;

    constructor(
        parameters: Parameter<Tensor>[],
        { lr = 6e-4, beta1 = 0.9, beta2 = 0.95, eps = 1e-8, weightDecay = 0 } = {}
    ) {
        super(parameters);
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
    }

    zeroGrad() {
        const ids = this.parameters.map(p => p.value._id);
        native.zeroGrad(ids);
    }

    step() {
        this.t++;
        const ids = this.parameters.map(p => p.value._id);
        native.adamwStep(ids, this.lr, this.beta1, this.beta2, this.eps, this.weightDecay, this.t);
    }
}
