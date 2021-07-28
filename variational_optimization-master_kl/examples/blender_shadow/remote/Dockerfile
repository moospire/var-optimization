# build: docker build . -t blender-shadow-worker:latest
FROM ubuntu:xenial
RUN apt-get update && \
      apt-get -y install sudo
RUN adduser --disabled-password --gecos '' docker && \
adduser docker sudo && \
echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER docker
WORKDIR /home/docker

RUN sudo apt-get update && \
	sudo apt-get install -y \
		curl \
		bzip2 \
		libfreetype6 \
		libgl1-mesa-dev \
		libglu1-mesa \
		libxi6 \
		libxrender1 \
        gcc && \
    sudo apt-get -y autoremove && \
    sudo rm -rf /var/lib/apt/lists/*

ENV BLENDER_MAJOR 2.79
ENV BLENDER_VERSION 2.79
ENV BLENDER_BZ2_URL https://mirror.clarkson.edu/blender/release/Blender$BLENDER_MAJOR/blender-$BLENDER_VERSION-linux-glibc219-x86_64.tar.bz2

RUN sudo mkdir /usr/local/blender && \
	curl -SL "$BLENDER_BZ2_URL" -o blender.tar.bz2 && \
	sudo tar -jxvf blender.tar.bz2 -C /usr/local/blender --strip-components=1 && \
	rm blender.tar.bz2

RUN curl -SL https://repo.anaconda.com/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh -o ~/miniconda.sh && \
/bin/bash ./miniconda.sh -b && \
. miniconda3/etc/profile.d/conda.sh && \
conda create -y --name blender python=3.5 && \
conda activate blender && \
pip install --upgrade pip && \
pip install redis==3.0.1 && \
pip install numpy==1.14.3 && \
pip install grpcio==1.17.1 && \
pip install grpcio-tools==1.22.0 && \
pip install pillow==5.4.1 && \
pip install scipy==1.2.0 && \
mv /home/docker/miniconda3/envs/blender/lib/python3.5/site-packages/redis /usr/local/blender/2.79/python/lib/python3.5/site-packages/redis && \
mv /home/docker/miniconda3/envs/blender/lib/python3.5/site-packages/numpy /usr/local/blender/2.79/python/lib/python3.5/site-packages/numpy && \
mv /home/docker/miniconda3/envs/blender/lib/python3.5/site-packages/grpc /usr/local/blender/2.79/python/lib/python3.5/site-packages/grpc && \
mv /home/docker/miniconda3/envs/blender/lib/python3.5/site-packages/google /usr/local/blender/2.79/python/lib/python3.5/site-packages/google && \
mv /home/docker/miniconda3/envs/blender/lib/python3.5/site-packages/PIL /usr/local/blender/2.79/python/lib/python3.5/site-packages/PIL && \
mv /home/docker/miniconda3/envs/blender/lib/python3.5/site-packages/scipy /usr/local/blender/2.79/python/lib/python3.5/site-packages/scipy && \
mv /home/docker/miniconda3/envs/blender/lib/python3.5/site-packages/six.py /usr/local/blender/2.79/python/lib/python3.5/site-packages/six.py && \
rm -rf /home/docker/miniconda3

EXPOSE 50051

ENTRYPOINT [ "/bin/bash","-c"]
