FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime 
# Note: Using a slightly more stable version for wider compatibility

# Setup user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user . .
RUN pip install --no-cache-dir -e .

# Permissions for entrypoint
RUN chmod +x entrypoint.sh

# HF uses 7860 by default
EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV KMC_ALGO_MODE=all

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]