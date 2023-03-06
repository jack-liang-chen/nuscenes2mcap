FROM ros:noetic-ros-core
RUN sed -i 's#http://archive.ubuntu.com/#http://mirrors.163.com/#' /etc/apt/sources.list && apt-get update

RUN apt-get install -y git python3-pip
RUN apt-get install -y git python3-tf2-ros
RUN apt-get install -y git ros-noetic-foxglove-msgs
RUN apt-get install -y git libgl1
RUN apt-get install -y git libgeos-dev

RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install pipenv
RUN pip3 install shapely==1.8.*
RUN pip3 install numpy==1.19.5
RUN pip3 install nuscenes-devkit
RUN pip3 install mcap
RUN pip3 install 'mcap-protobuf-support>=0.0.8'
RUN pip3 install foxglove-data-platform
RUN pip3 install tqdm
RUN pip3 install requests
RUN pip3 install protobuf

RUN pip3 install git+https://github.com/DanielPollithy/pypcd.git

COPY . /work

WORKDIR /work
