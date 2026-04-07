#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr};
#[cfg(feature = "cuda")]
use cudarc::cublas::safe::CudaBlas;
#[cfg(feature = "cuda")]
use cudarc::nvrtc;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "cuda")]
pub struct GpuDevice {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    functions: HashMap<String, CudaFunction>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for GpuDevice {}
#[cfg(feature = "cuda")]
unsafe impl Sync for GpuDevice {}

#[cfg(feature = "cuda")]
static GPU: OnceLock<GpuDevice> = OnceLock::new();

#[cfg(feature = "cuda")]
impl GpuDevice {
    pub fn instance() -> &'static Self {
        GPU.get_or_init(Self::init)
    }

    fn init() -> Self {
        let ctx = CudaContext::new(0).expect("CUDA context creation failed");
        unsafe { ctx.disable_event_tracking(); }
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).expect("cuBLAS handle creation failed");

        let mut dev = Self {
            ctx,
            stream,
            blas,
            functions: HashMap::new(),
        };
        dev.load_all_kernels();
        dev
    }

    fn compile_and_load(&mut self, name: &str, source: &str, kernel_names: &[&str]) {
        let ptx = nvrtc::compile_ptx(source)
            .unwrap_or_else(|e| panic!("Failed to compile {name}: {e}"));
        let module = self
            .ctx
            .load_module(ptx)
            .unwrap_or_else(|e| panic!("Failed to load module {name}: {e}"));
        for &kname in kernel_names {
            let func = module
                .load_function(kname)
                .unwrap_or_else(|e| panic!("Failed to get function {kname}: {e}"));
            self.functions.insert(kname.to_string(), func);
        }
    }

    fn load_all_kernels(&mut self) {
        self.compile_and_load(
            "elementwise",
            include_str!("../kernels/elementwise.cu"),
            &[
                "add_f32",
                "sub_f32",
                "mul_f32",
                "neg_f32",
                "mul_scalar_f32",
                "exp_f32",
                "log_f32",
                "add_bias_f32",
                "fill_f32",
                "div_f32",
                "copy_f32",
                "permute_f32",
                "broadcast_add_f32",
                "broadcast_mul_f32",
                "sum_reduce_all_f32",
            ],
        );
        self.compile_and_load(
            "activation",
            include_str!("../kernels/activation.cu"),
            &[
                "gelu_forward_f32",
                "gelu_backward_f32",
                "relu_forward_f32",
                "relu_backward_f32",
            ],
        );
        self.compile_and_load(
            "reduce",
            include_str!("../kernels/reduce.cu"),
            &[
                "sum_along_dim_f32",
                "mean_along_dim_f32",
                "max_along_dim_f32",
                "sum_broadcast_f32",
            ],
        );
        self.compile_and_load(
            "layernorm",
            include_str!("../kernels/layernorm.cu"),
            &["layernorm_forward_f32", "layernorm_backward_f32"],
        );
        self.compile_and_load(
            "softmax",
            include_str!("../kernels/softmax.cu"),
            &["softmax_forward_f32", "softmax_backward_f32"],
        );
        self.compile_and_load(
            "cross_entropy",
            include_str!("../kernels/cross_entropy.cu"),
            &["cross_entropy_forward_f32", "cross_entropy_backward_f32"],
        );
        self.compile_and_load(
            "embedding",
            include_str!("../kernels/embedding.cu"),
            &["embedding_forward_f32", "embedding_backward_f32"],
        );
        self.compile_and_load(
            "dropout",
            include_str!("../kernels/dropout.cu"),
            &["dropout_apply_f32", "dropout_backward_f32"],
        );
        self.compile_and_load(
            "adamw",
            include_str!("../kernels/adamw.cu"),
            &["adamw_step_f32"],
        );
        self.compile_and_load(
            "grad_util",
            include_str!("../kernels/grad_util.cu"),
            &["grad_norm_sq_partial_f32", "grad_clip_f32"],
        );
        self.compile_and_load(
            "data",
            include_str!("../kernels/data.cu"),
            &["sample_batch_i32"],
        );
        self.compile_and_load(
            "flash_attention",
            include_str!("../kernels/flash_attention.cu"),
            &["flash_attention_forward_f32", "flash_attention_backward_f32"],
        );
        self.compile_and_load(
            "fused_ops",
            include_str!("../kernels/fused_ops.cu"),
            &["residual_layernorm_forward_f32", "bias_gelu_forward_f32", "bias_gelu_backward_f32"],
        );
        self.compile_and_load(
            "mixed_precision",
            include_str!("../kernels/mixed_precision.cu"),
            &["f32_to_bf16", "bf16_to_f32", "scale_f32", "check_inf_nan_f32"],
        );
    }

    pub fn get_func(&self, name: &str) -> &CudaFunction {
        self.functions
            .get(name)
            .unwrap_or_else(|| panic!("Kernel function '{name}' not found"))
    }

    pub fn ptr<T>(&self, slice: &CudaSlice<T>) -> u64
    where
        CudaSlice<T>: DevicePtr<T>,
    {
        let (ptr, _guard) = slice.device_ptr(&self.stream);
        ptr
    }
}

#[cfg(feature = "cpu")]
pub struct GpuDevice;

#[cfg(feature = "cpu")]
impl GpuDevice {
    pub fn instance() -> &'static Self {
        static STUB: GpuDevice = GpuDevice;
        &STUB
    }
}
