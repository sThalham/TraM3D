#!/bin/bash

docker build --no-cache -t tram6d_gpu_0 .
thispid=$(docker run --gpus '"device=0"' --network=host --shm-size=8gb --name=tram6d_gpu_0 -t -d -v /hdd/template_pose_data:/TraM6D/template_data tram6d_gpu_0)
