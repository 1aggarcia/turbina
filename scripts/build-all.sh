#! /bin/bash

cargo fmt
cargo build -r
rm -rf /usr/local/bin/turbina
cp target/release/turbina /usr/local/bin/turbina

./build-wasm.sh
cd online-playground
pnpm run build
