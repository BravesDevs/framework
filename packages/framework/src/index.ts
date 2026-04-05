export { Tensor, native, type TensorLike, type Shape } from "./tensor.js";
export { Module, Parameter } from "./module.js";
export {
    Linear, ReLU, Sigmoid, Tanh, Embedding,
    softmax, logsoftmax, gelu, dropout,
    crossEntropyLoss, mseLoss, layerNorm,
    randRange, tile, avgpool2d, maxpool2d,
} from "./nn.js";
export { Optimizer, SGD, Adam, type ParameterValue } from "./optimizer.js";

export function destroyPool(): void { /* no-op in native backend */ }
export function destroyDevice(): void { /* no-op in native backend */ }
