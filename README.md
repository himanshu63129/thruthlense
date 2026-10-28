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

## 1. Train the model (once, about 10 minutes on a free Colab GPU)

The model weights are not in the repo (about 100 MB), so train them first.

1. Open [Google Colab](https://colab.research.google.com), choose **Runtime > Change runtime type > T4 GPU**.
2. Upload these three files from `ai-service/` to the Colab file panel: `train.py`, `arch.py`, `requirements-train.txt`.
3. Run in a cell:
   ```python
   !pip install -q datasets pillow numpy
   !python train.py --samples 12000 --epochs 3
   ```
   The script prints how it mapped the dataset labels (check that the AI class maps to `ai_generated`).
   If it says it cannot tell, pass `--ai-label-index 0` or `1`.
4. Download `models/truthlens_best.pth` and the `samples/` folder from Colab.
5. Put the weights at `ai-service/models/truthlens_best.pth` and keep `samples/` for the demo.

More accuracy: raise `--samples` (for example 30000) and `--epochs 5`. More images means a longer download and more RAM.

## 2. Run it (three terminals)

```bash
# Terminal 1: AI service
cd ai-service
python -m venv venv && venv\Scripts\activate      # Windows. On Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python main.py                                      # http://127.0.0.1:8000/health

# Terminal 2: API server
cd server
npm install
copy .env.example .env                              # Mac/Linux: cp .env.example .env
npm run dev                                         # http://localhost:5000/api/health

# Terminal 3: website
cd client
npm install
npm run dev                                         # http://localhost:5173
```

MongoDB is optional. Without it, detection still works; results just are not saved.

The status light in the website header tells you the state: green means the trained model is loaded.
Amber means no weights were found and scores come from a placeholder heuristic, so do not present those as real results.

## 3. Demo tips

- Use the images in `ai-service/samples/` (held-out images from the dataset, 6 real and 6 AI) for a reliable demo.
- The model learns the artifacts of the generators in its training data. Photos from newer generators, or heavily compressed or screenshotted images, can fool it. Say so if asked: the score is a probability, not proof.

## API

`POST /api/detect/image` with form field `file` (JPG, PNG or WEBP, up to 25 MB) returns:

```json
{ "prediction": "authentic | ai_generated | inconclusive", "score": 0.87, "confidenceLevel": "high",
  "processingTimeMs": 310, "forensicMetrics": { "resolution": "1024x1024", "modelWeightsActive": true } }
```

`score` is the probability the image is AI-generated. Below 0.35 is authentic, above 0.65 is AI-generated, and between is inconclusive.
Video screening exists in the API (`/api/detect/video`) but the website only handles images.
