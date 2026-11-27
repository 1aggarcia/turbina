# Compile turbina for x86-64 on an ARM machine using a Docker container

docker buildx build --platform linux/amd64 --output type=local,dest=binaries/x86_64 .
