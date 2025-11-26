# Face Attendance (Production-Grade Skeleton)

Real-time face-recognition attendance with FastAPI (backend), InsightFace (ArcFace), Qdrant (vector search), Postgres (records), and a Next.js frontend.

## Quick start (Docker)
1) Copy `.env.example` to `.env` inside `backend/` and fill values.
2) `docker compose up --build`
3) Backend docs: http://localhost:8000/docs

### Non-Docker (dev)
- Python 3.11+, Node 20+
- `cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload`

## Endpoints
- `POST /enroll` (multipart: image + user_id)
- `WS   /match/stream` (send base64 frames → match reply)
- `GET  /attendance/export?date=today` (xlsx)

> This repo is scaffolded to be readable and extendable. Tweak thresholds in `routers/match.py`.
