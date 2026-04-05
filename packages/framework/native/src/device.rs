/// Device abstraction. Currently CPU-only; CUDA variant behind feature flag.
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(CudaState),
}

#[cfg(feature = "cuda")]
pub struct CudaState {
    pub device: cudarc::driver::CudaDevice,
    pub blas: cudarc::cublas::CudaBlas,
}
