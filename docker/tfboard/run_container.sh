#!/bin/bash

dockerLogDir=/diffusion_logs
hostLogDir=/home/preciselyv/nsu/diploma/diffusionDirs/results/logs
img=diffusion-tfboard

docker run -d --name diffusion_Monitor -p 6007:6007 \
           -v $hostLogDir:$dockerLogDir \
           $img \
           tensorboard --logdir $dockerLogDir \
           --port 6007 --bind_all 
	   
