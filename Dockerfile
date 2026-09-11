# CPU-only image: the app loads the trained artifact and never retrains.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=2
WORKDIR /srv/ams

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ams/ ./ams/
COPY app/ ./app/
COPY models/ ./models/
COPY samples/ ./samples/
COPY reports/ ./reports/
COPY configs/ ./configs/
COPY train.py ./

EXPOSE 8000
# AMS_MODELS_DIR etc. can be overridden at deploy time.
CMD ["python", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
