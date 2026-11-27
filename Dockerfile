# This Dockerfile is for compiling Turbina for CPU architectures different than
# the developer's host architecture to run Turbina programs in containerized
# applications (e.g ARM-based Mac compiling for x86-64)

# How to run (example is for x86-64)
# docker buildx build --platform linux/amd64 --output type=local,dest=binaries/x86_64 .

FROM rust:bookworm AS builder
WORKDIR /app

COPY Cargo.toml Cargo.lock .
COPY src ./src

RUN cargo build --release

FROM scratch
COPY --from=builder /app/target/release/turbina /turbina
