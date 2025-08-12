# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Ensure non-interactive frontend for APT
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    tzdata

# Install Go
RUN curl -OL https://golang.org/dl/go1.22.4.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.22.4.linux-amd64.tar.gz && \
    rm go1.22.4.linux-amd64.tar.gz

# Set Go environment variables
ENV PATH="/usr/local/go/bin:${PATH}"
ENV GO111MODULE=on

# Clone Cosmos SDK and build simd
RUN git clone https://github.com/cosmos/cosmos-sdk /cosmos-sdk && \
    cd /cosmos-sdk && \
    git checkout v0.46.0 && \
    make build && \
    cp /cosmos-sdk/build/simd /usr/local/bin/simd

# Copy scripts and configs
COPY localnet.ps1 /localnet.ps1
COPY init.sh /init.sh
COPY node0/config /node0/config
COPY node1/config /node1/config

# Expose ports for p2p and RPC
EXPOSE 26656 26657

# Set the default command to run the initialization script
CMD ["/init.sh"]
