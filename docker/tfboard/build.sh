#!/bin/bash

img=diffusion-tfboard

docker build -t $img -f ./Dockerfile .
