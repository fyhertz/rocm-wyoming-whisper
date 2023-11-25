# rocm-wyoming-whisper

A docker image and a few lines of python to use OpenAI whisper with Rhasspy and/or Home Assistant on AMD GPUs with ROCm.

## Run with docker-compose

```shell
docker-compose up -d
```

## Run with Docker

Build docker image:

```shell
docker build -t wyoming-whisper .
```

Run docker image:

```shell
docker run --entrypoint '' -v $(pwd)/data:/data -v $(pwd)/src:/src -it --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video -p 10300:10300 wyoming-whisper bash
```

Run script:

```shell
python -m wyoming_whisper --download-dir /data --model medium --debug
```
