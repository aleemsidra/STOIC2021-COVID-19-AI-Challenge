FROM nvcr.io/nvidia/pytorch:21.09-py3
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y
RUN apt install -y \
wget \
unzip
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /image_classification_pytorch
RUN pip install tensorflow wandb tensorboardX scipy pandas json2html cockpit-for-pytorch easydict SimpleITK
RUN pip install --upgrade numpy
RUN apt install -y tmux