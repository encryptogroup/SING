FROM ubuntu:22.04

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get update -qy && apt-get install -y \
curl \
cmake \
coinor-cbc \
coinor-libcbc-dev \
cvc4 \
git \
libgmp-dev \
libboost-all-dev \
libssl-dev \
time \
zsh \
build-essential \
python3 \
python3-pip

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY . /circ/

WORKDIR /circ
RUN python3 driver.py --all_features
RUN mkdir -p scripts/aby_tests/tests
RUN python3 driver.py -c
RUN python3 driver.py -i
RUN python3 driver.py -b
