# Stage 1: Build frontend
FROM node:20-alpine AS frontend
WORKDIR /app/web/immersive
COPY web/immersive/package.json web/immersive/package-lock.json ./
RUN npm ci
COPY web/immersive/ ./
ENV VITE_API_URL=relative
RUN npm run build

# Stage 2: Runtime (FastAPI + static assets)
FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY web/api/requirements.txt /app/web/api/requirements.txt
RUN pip install --no-cache-dir -r /app/web/api/requirements.txt && \
    rm -rf /root/.cache/pip /tmp/*

COPY web/api /app/web/api
COPY cascades /app/cascades
COPY second_phase /app/second_phase
COPY configs /app/configs
COPY data/raw/prices_1h_730d.csv /app/data/raw/prices_1h_730d.csv
COPY artifacts /app/artifacts
COPY --from=frontend /app/web/immersive/dist /app/web/immersive/dist

EXPOSE 7860
CMD ["uvicorn", "main:app", "--app-dir", "web/api", "--host", "0.0.0.0", "--port", "7860"]
