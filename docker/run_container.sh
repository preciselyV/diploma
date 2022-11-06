
cname=th-diffusion
docker_image=diffusion-talking-head
result_path=~/nsu/diploma/results

docker run --gpus all -it --rm --name $cname \
            -v $result_path:/results \
            $docker_image
