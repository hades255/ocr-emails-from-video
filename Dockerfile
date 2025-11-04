FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git ffmpeg \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt constraints.txt ./

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip install --force-reinstall "numpy==1.26.4" && \
    pip install -r requirements.txt -c constraints.txt

COPY . .

CMD ["python3", "main.py"]
