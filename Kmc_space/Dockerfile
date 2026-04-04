FROM python:3.11-slim

RUN useradd -m -u 1000 kmc
USER kmc
ENV PATH="/home/kmc/.local/bin:$PATH"

WORKDIR /app

COPY --chown=kmc requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=kmc . /app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]