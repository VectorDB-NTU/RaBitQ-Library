
extern crate bindgen;
extern crate cc;

use std::env;
use std::path::PathBuf;

fn main() {
    // Compile the C++ wrapper.
    cc::Build::new()
        .file("rabitq_wrapper.cpp")
        .include("../../rabitqlib") 
        .cpp(true)
        .flag("-std=c++17")
        .flag("-march=native")
        .flag("-mavx512f")
        .flag("-fopenmp")
        .compile("rabitq_wrapper");

    println!("cargo:rustc-link-lib=stdc++");

    // Generate bindings for the C header.
    let bindings = bindgen::Builder::default()
        .header("rabitq.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}