# add your Docker build steps here
FROM python:3.11-slim

# System deps for opencv + insightface
RUN apt-get update && apt-get install -y \
    build-essential git libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Speed up installs
ENV PIP_NO_CACHE_DIR=1
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# App
COPY app /app/app

# InsightFace will download models on first run; you can pre-warm via:
# RUN python -c "from insightface.app import FaceAnalysis; a=FaceAnalysis(name='buffalo_l'); a.prepare(ctx_id=0, det_size=(640,640))"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host","0.0.0.0","--port","8000","--workers","2"]
