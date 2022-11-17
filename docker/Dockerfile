FROM nvcr.io/nvidia/tensorrt:22.10-py3

RUN pip3 install torch torchvision torchaudio pandas numpy scikit-learn tqdm pyyaml

COPY . /diffusion-th

WORKDIR /diffusion-th
