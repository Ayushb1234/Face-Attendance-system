## 🎭 Face Attendance System (AI-Powered | FastAPI + React)

A production-ready real-time face attendance automation system built using **Python, OpenCV, InsightFace Embeddings & Vector Similarity Search** for identity recognition — paired with a modern **React.js frontend**.

This system automatically detects faces, assigns identity labels, stores new faces, and marks attendance—including duplicate check prevention.

---

## 🚀 Features

| Feature | Description |
|--------|------------|
| 🔍 Real-time face detection | Uses OpenCV + InsightFace for fast, lightweight face tracking |
| 🧠 Face recognition | Generates embeddings & matches faces using cosine similarity |
| ➕ Auto registration | New faces are automatically stored without manual entry |
| 📁 Attendance storage | Saves attendance logs in CSV/Excel format |
| 🖥️ Frontend UI | Built with React.js for live camera preview & status updates |
| 🏗️ Scalable design | Modular ML pipeline and REST API |


---

## 🏗️ Tech Stack

### **Backend**
- FastAPI
- Python 3.x
- InsightFace / FaceNet embeddings
- OpenCV
- NumPy
- Pandas
- FAISS / cosine similarity search (optional)

### **Frontend**
- React.js
- Axios
- Webcam / MediaPipe API
- TailwindCSS (optional)

---

## 📂 Project Structure



Screenshots of the Project:
--------------------------
<img width="1127" height="371" alt="image" src="https://github.com/user-attachments/assets/5cbc5ae9-9908-46a5-ad53-5e78213dfaa0" />
<img width="722" height="410" alt="image" src="https://github.com/user-attachments/assets/ace8af07-e5dc-4b5f-aa9f-7329b649492a" />


---

## 📂 Project Structure
```
.
├── app/ # Backend FastAPI service
├── data/ # Stored embeddings / attendance logs
├── dataset/ # Face dataset (generated automatically)
├── scripts/ # Helper utilities (training, embedding export)
├── tests/ # API + model tests
├── frontend/ # React.js UI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

```yaml
Copy code

---

## ⚙️ Installation & Setup
```

### 1️⃣ Clone Repo

```bash
git clone https://github.com/<your-username>/face-attendance-system.git
cd face-attendance-system
```
2️⃣ Backend Setup (FastAPI)

```bash
Copy code
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
Backend will run at:

cpp
Copy code
http://127.0.0.1:8000
Swagger Docs:

arduino
Copy code
http://127.0.0.1:8000/docs
```
3️⃣ Frontend Setup (React)

```bash
Copy code
cd ../frontend
npm install
npm run dev
Frontend runs at:

arduino
Copy code
http://localhost:5173
```

🎯 How It Works
----------------

User allows webcam access on frontend

React records frames → sends them to backend

Backend extracts face embeddings and:

Matches existing user → attendance auto-marked

If no match → new user enrolled

Attendance saved in .csv files + timestamp

📊 Attendance Format Example
------------------------------
```pgsql
Copy code
Name, Date, Time
Ayush, 2025-02-01, 09:41:32
Rahul, 2025-02-01, 10:05:17
🧪 Testing
Run backend unit tests:
```

bash
Copy code
pytest
📦 Docker Support
bash
Copy code
docker-compose up --build
🛠️ Future Enhancements
📱 Mobile app support

🔐 Role-based authentication

🌐 Cloud storage (S3, Firebase, GSheets)

🧠 Liveness detection & anti-spoofing

📈 Attendance analytics dashboard

🤝 Contributing
Pull requests are welcome!
For major changes, please open an issue first.

📄 License
MIT License

👤 Author
Ayush Choudhary
🚀 AI/ML Developer | Computer Vision Engineer
🔗 GitHub · LinkedIn · Portfolio



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
