FROM nvidia/cuda:8.0-devel-ubuntu16.04

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get update
RUN apt-get install -y wget locales cmake libopenblas-dev libgtest-dev
RUN locale-gen en_US.UTF-8
RUN cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib
