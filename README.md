# Face Attendance (Production-Grade Skeleton)

Real-time face-recognition attendance with FastAPI (backend), InsightFace (ArcFace), Qdrant (vector search), Postgres (records), and a Next.js frontend.

Screenshots of the Project:
--------------------------
<img width="1127" height="371" alt="image" src="https://github.com/user-attachments/assets/5cbc5ae9-9908-46a5-ad53-5e78213dfaa0" />
<img width="722" height="410" alt="image" src="https://github.com/user-attachments/assets/ace8af07-e5dc-4b5f-aa9f-7329b649492a" />


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
