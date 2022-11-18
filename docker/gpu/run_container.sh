
cname=th-diffusion
docker_image=diffusion-talking-head
checkpoints_path=~/nsu/diploma/diffusionDirs/results/checkpoints
logs_path=~/nsu/diploma/diffusionDirs/results/logs
configs_path=~/nsu/diploma/diffusionDirs/confs
datasets_path=~/nsu/diploma/diffusionDirs/datasets


docker run --gpus all -it --rm --name $cname \
            -v $configs_path:/diffusion-th/configs \
            -v $checkpoints_path:/diffusion-th/results/checkpoints \
            -v $logs_path:/diffusion-th/results/logs \
            -v $datasets_path:/diffusion-th/datasets \
            $docker_image
