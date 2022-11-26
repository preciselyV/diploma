FROM ubuntu:18.04

RUN apt update && apt install -y python3.8 python3.8-distutils curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
RUN python3.8 /tmp/get-pip.py

RUN pip3.8 install tensorboard
