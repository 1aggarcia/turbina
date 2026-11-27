#! /bin/bash

cargo fmt
cargo build -r
rm -rf /usr/local/bin/turbina
cp target/release/turbina /usr/local/bin/turbina

./scripts/build-wasm.sh
cd online-playground
pnpm run build
