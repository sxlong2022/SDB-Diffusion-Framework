# hash:sha256:920270ee3191a04e514301dcd4bbfd8de916cb45fa587310261017ced4f217a8
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U --no-cache-dir \
    einops==0.8.0 \
    jax[cuda]==0.4.13 \
    jaxlib==0.4.13 \
    jupyter==1.0.0 \
    matplotlib==3.7.5 \
    ml-collections==0.1.0 \
    numpy==1.24.4 \
    pandas==1.4.2 \
    pillow==9.2.0 \
    requests==2.27.1 \
    runipy==0.1.5 \
    scipy==1.10.1 \
    six==1.15.0 \
    spicy==0.16.0 \
    tensorflow==2.13.1 \
    torch==1.12.1 \
    torchaudio==0.12.1 \
    torchvision==0.13.1 \
    tqdm==4.66.2

COPY postInstall /
RUN /postInstall
