#! /bin/bash
# if [-n "$1"]
# then
#     echo "The built yoho_dev image is large (around 1GB). "
#     echo "Build docker images for yoho_dev object detection..."
#     # --squash: Squash newly built layers into a single new layer
#     # Used to reduce built image size.
#     sudo docker build -t yoho_dev:1 -f ./Dockerfile2.yoho_dev .
#     sudo docker image prune --force
# else 
#     echo "The built yoho_dev image is large (around 1GB). "
#     echo "Build version speical for [China] !"
#     echo "Build docker images for yoho_dev object detection..."
#     # --squash: Squash newly built layers into a single new layer
#     # Used to reduce built image size.
#     sudo docker build -t yoho_dev:1 -f ./Dockerfile.yoho_dev .
#     sudo docker image prune --force

# fi

echo "The built yoho_dev image is large (around 1GB). "
# echo "Build version speical for [China] !"
echo "Build docker images for yoho_dev object detection..."
# --squash: Squash newly built layers into a single new layer
# Used to reduce built image size.
sudo docker build -t yoho_dev:1 -f ./Dockerfile.yoho_dev .
sudo docker image prune --force

