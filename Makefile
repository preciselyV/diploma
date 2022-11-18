build:
	echo "bulding container"
	docker build -t diffusion-talking-head -f docker/gpu/Dockerfile .

run:
	bash -x docker/gpu/run_container.sh





