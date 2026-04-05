FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# 1. Sabse pehle ROOT bano taaki system install ho sake
USER root

# 2. System dependencies install karein
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# 3. User create karein
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# 4. Requirements copy aur install karein (Root hi rehne dein temporarily)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Saara code copy karein aur permissions set karein
COPY . .
RUN pip install --no-cache-dir -e .
RUN chmod +x entrypoint.sh && chown -R user:user $HOME/app

# 6. Ab vapas HF ke 'user' par switch karein (Security ke liye)
USER user

# 7. Port aur Env settings
EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    KMC_ALGO_MODE=all

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]