FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libegl1-mesa-dev libgl1-mesa-dri libosmesa6-dev \
    libglib2.0-0 libsndfile1 wget unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Rhubarb Lip Sync
RUN wget -q https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/Rhubarb-Lip-Sync-1.13.0-Linux.zip \
    && unzip Rhubarb-Lip-Sync-1.13.0-Linux.zip -d /tmp/rhubarb \
    && mv /tmp/rhubarb/rhubarb /usr/local/bin/ \
    && rm -rf Rhubarb-Lip-Sync-1.13.0-Linux.zip /tmp/rhubarb

WORKDIR /app

COPY pyproject.toml .
COPY clawvatar/ clawvatar/

RUN pip install --no-cache-dir .

VOLUME /app/avatars

EXPOSE 8765

ENTRYPOINT ["clawvatar", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8765"]
