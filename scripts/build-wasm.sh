#! /bin/bash

# compiles Turbina to a Web Assembly package in the `online-playground` directory
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' wasm-pack build --target web --out-dir online-playground/pkg
