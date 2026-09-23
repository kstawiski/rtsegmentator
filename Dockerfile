FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORTAL_DATA=/data \
    WORKER_MODE=local \
    LOCAL_PYTHON=/opt/rtsegmentator/venv/bin/python \
    LOCAL_RUNNER=/app/run_task.py

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip python3-dev build-essential openssh-server rsync ca-certificates git && \
    rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd /data /models /runtimes

WORKDIR /app
COPY requirements.txt requirements-worker.txt constraints-worker.txt ./
RUN python3 -m venv /opt/rtsegmentator/venv && \
    /opt/rtsegmentator/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/rtsegmentator/venv/bin/pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 && \
    /opt/rtsegmentator/venv/bin/pip install --no-cache-dir -c constraints-worker.txt -r requirements-worker.txt

COPY . /app
RUN chmod +x /app/docker/entrypoint.sh && \
    useradd -m -s /bin/bash worker && mkdir -p /home/worker/jobs /home/worker/.ssh && \
    chown -R worker:worker /home/worker

EXPOSE 8080 2222
VOLUME ["/data", "/models", "/runtimes"]
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["portal"]
