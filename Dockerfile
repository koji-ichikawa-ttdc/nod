FROM python:3.10

ENV TZ Asia/Tokyo
RUN echo "${TZ}" > /etc/timezone && \
    cp /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update -y && apt-get install -y \
    libgl1-mesa-glx libgl1-mesa-dev && \
#cash削除
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/*

WORKDIR /nod
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt && \
    # pip install -v -e /workspaces/nod/YOLOX && \
#cash削除
    rm -rf ~/.cache/pip/*
