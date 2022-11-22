#!/bin/bash

img=tfboard

docker build -t $img -f ./Dockerfile .
