build:
	echo "bulding container"
	docker build -t diffusion-talking-head -f docker/Dockerfile .
run:
	bash -x docker/run_container.sh





