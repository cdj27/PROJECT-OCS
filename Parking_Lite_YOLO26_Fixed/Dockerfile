FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PARKING_CPU_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib YOLO_CONFIG_DIR=/tmp/ultralytics
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements-hosted.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-hosted.txt
RUN useradd --create-home --uid 1000 parking
COPY --chown=parking:parking parking_lite ./parking_lite
COPY --chown=parking:parking hosted_api.py ./hosted_api.py
COPY --chown=parking:parking deploy/verify_models.py ./deploy/verify_models.py
COPY --chown=parking:parking deploy/entrypoint.sh ./deploy/entrypoint.sh
USER parking
EXPOSE 8080
ENTRYPOINT ["sh","/app/deploy/entrypoint.sh"]
CMD ["python","-m","uvicorn","hosted_api:create_app","--factory","--host","0.0.0.0","--port","8080","--workers","1","--limit-concurrency","8"]
