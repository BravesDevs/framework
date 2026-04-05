extern crate napi_build;

fn main() {
    napi_build::setup();

    #[cfg(feature = "cuda")]
    compile_ptx_kernels();
}

#[cfg(feature = "cuda")]
fn compile_ptx_kernels() {
    use std::process::Command;
    use std::path::Path;
    use std::fs;

    let kernel_dir = Path::new("kernels");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_dir = Path::new(&out_dir).join("ptx");
    fs::create_dir_all(&ptx_dir).unwrap();

    let cu_files: Vec<_> = fs::read_dir(kernel_dir)
        .expect("kernels/ directory not found")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "cu"))
        .collect();

    for entry in &cu_files {
        let src = entry.path();
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let ptx_path = ptx_dir.join(format!("{}.ptx", stem));

        let status = Command::new("nvcc")
            .args([
                "--ptx",
                "-O3",
                "-arch=sm_80",
                "-o", ptx_path.to_str().unwrap(),
                src.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to run nvcc. Is the CUDA toolkit installed?");

        assert!(status.success(), "nvcc failed for {}", src.display());
        println!("cargo:rerun-if-changed={}", src.display());
    }
}
