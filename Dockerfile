FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

# install dinfer in editable mode
RUN pip install --no-cache-dir -e .

# install all other packages
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "server/llada_server.py"]