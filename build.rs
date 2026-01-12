fn main() {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("src/umappp_bridge.cpp")
        .include("vendor/umappp/include")
        .include("vendor/knncolle/include")
        .include("vendor/aarand/include")
        .include("vendor/subpar/include")
        .include("vendor/sanisizer/include")
        .include("vendor/irlba/include")
        .include("vendor/eigen")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .flag_if_supported("-pthread");

    build.compile("umappp_ffi");

    if cfg!(target_family = "unix") {
        println!("cargo:rustc-link-lib=pthread");
    }

    println!("cargo:rerun-if-changed=src/umappp_bridge.cpp");
    println!("cargo:rerun-if-changed=src/umappp_bridge.h");
    println!("cargo:rerun-if-changed=vendor/umappp/include");
    println!("cargo:rerun-if-changed=vendor/knncolle/include");
    println!("cargo:rerun-if-changed=vendor/aarand/include");
    println!("cargo:rerun-if-changed=vendor/subpar/include");
    println!("cargo:rerun-if-changed=vendor/sanisizer/include");
    println!("cargo:rerun-if-changed=vendor/irlba/include");
    println!("cargo:rerun-if-changed=vendor/eigen/Eigen");
}
