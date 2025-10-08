FROM python:3.11

COPY . /workspace
WORKDIR /workspace

RUN pip install -r requirements.txt
RUN pip install matplotlib
RUN pip install "ray[rllib]==2.49.2"
RUN pip install torch
RUN pip install pygame

ENTRYPOINT ["python3"]