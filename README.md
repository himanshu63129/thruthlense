# TruthLens

Upload a photo and TruthLens estimates whether it is a real photograph or AI-generated.

```
React client (Vite, :5173)  ->  Express API (:5000)  ->  FastAPI + PyTorch model (:8000)
                                    |
                                MongoDB (optional: stores history, auth, reports)
```

| Folder | What it does |
| --- | --- |
| `client/` | Upload page: drag and drop, scan animation, verdict and score meter |
| `server/` | Express API: validates uploads, calls the AI service, saves results if MongoDB is running |
| `ai-service/` | FastAPI service running a ResNet50 classifier; `train.py` trains it |

## Quickstart

Start the three services:
1. `cd ai-service && uvicorn main:app --port 8000`
2. `cd server && npm start`
3. `cd client && npm run dev`
