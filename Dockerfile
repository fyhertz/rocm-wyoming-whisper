FROM rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1

RUN pip install -U pip && pip install wyoming==1.2.0 openai-whisper==20231117 tokenizers==0.13.*

COPY src /src

WORKDIR /src

ENTRYPOINT ["/src/run.sh"]
