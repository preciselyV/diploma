
cname=th-diffusion
docker_image=diffusion-talking-head

docker run --gpus all -it --rm --name $cname $docker_image
