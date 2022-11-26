FROM nvcr.io/nvidia/tensorflow:19.12-tf2-py3

RUN apt update && apt install -y python3.8
RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
RUN python3.8 /tmp/get-pip.py

RUN pip3.8 install torch==1.12.1+cu102 torchvision==0.13.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
RUN pip3.8 install pandas numpy scikit-learn tqdm pyyaml tensorboard
RUN pip3.8 install torch-fidelity torchmetrics

COPY . /diffusion-th

WORKDIR /diffusion-th
