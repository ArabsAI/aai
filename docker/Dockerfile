FROM ghcr.io/nvidia/jax:nightly-2023-10-25-linux-amd64 as base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y git openssh-client make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev ca-certificates vim gcc


RUN mkdir -p -m 0600  ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts


RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y

ENV GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV CLOUDSDK_PYTHON=/usr/bin/python3

WORKDIR /aai

RUN uv sync
