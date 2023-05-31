#!/bin/bash

docker build --no-cache -t tram3d_gpu_0 .
thispid=$(docker run --gpus '"device=0"' --network=host --shm-size=8gb --name=tram3d_gpu_0 -t -d -v /hdd/template_pose_data:/TraM3D/template_data tram3d_gpu_0)
