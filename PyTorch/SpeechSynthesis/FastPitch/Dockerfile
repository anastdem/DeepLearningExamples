ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.08-py3
FROM ${FROM_IMAGE_NAME}

ENV PYTHONPATH /workspace/fastpitch
WORKDIR /workspace/fastpitch

ADD requirements.txt .
RUN pip install -r requirements.txt

COPY . .
