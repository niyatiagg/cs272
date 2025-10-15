FROM python:3.11

COPY . /workspace
WORKDIR /workspace

RUN mkdir -p /workspace/logs
RUN mkdir -p /workspace/models

RUN pip install matplotlib
RUN pip install torch==2.8.0
RUN pip install 'stable-baselines3[extra]==2.7.0'

ENTRYPOINT ["python3"]