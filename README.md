# Emotion Explorer

A full workflow for multi-label emotion detection using Hugging Face transformers. The project includes:

- **Training pipeline** (`main.py`) that fine-tunes a sequence classifier on seven emotions (admiration, amusement, gratitude, love, pride, relief, remorse) using the provided `train.csv` / `dev.csv`.
- **Reusable inference module** (`emotion_inference.py`) plus shared config (`emotion_config.py`).
- **FastAPI + Jinja frontend** (`app/`) that lets you test the trained model via a simple UI, with Docker support for easy deployment.

## Quick Start

1. Install dependencies (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model (saves artifacts under `artifacts/`):
   ```bash
   python main.py
   ```
   Adjust common flags like `--model-name`, `--val-size`, `--output-dir` etc. with `python main.py --help`.
3. Launch the web app locally (after training):
   ```bash
   uvicorn app.server:app --reload
   ```
   Visit [http://localhost:8000](http://localhost:8000) to test inputs, toggle dark mode, and inspect probabilities. The API also exposes `POST /api/predict` for programmatic use.

## Docker

You can containerize the app once artifacts exist on the host:

```bash
docker build -t emotion-ui .
docker run --rm -p 8000:8000 -v "$PWD/artifacts:/app/artifacts" emotion-ui
```

The bind mount ensures the container sees your trained model. Set `ARTIFACT_DIR` if you stored artifacts elsewhere.

## Project Structure

```
main.py                # training CLI (HF Trainer + metrics + artifact saving)
emotion_config.py      # shared constants / defaults
emotion_inference.py   # EmotionPredictor helper used by CLI + API
app/
  server.py            # FastAPI app + REST endpoints
  templates/index.html # Jinja template for UI
  static/style.css     # Light/dark theme styles
Dockerfile             # container image for serving the UI
.dockerignore
train.csv, dev.csv     # provided datasets
requirements.txt
```

## Notes

- `main.py` automatically disables TensorFlow (`TRANSFORMERS_NO_TF=1`) for lean PyTorch setups.
- Training artifacts are stored as:
  - `artifacts/model/` – tokenizer + HF weights
  - `artifacts/thresholds.json` – per-label thresholds tuned on the validation split
  - `artifacts/config.json` – training configuration snapshot
- Re-run `python main.py` whenever you tweak hyperparameters or want to regenerate artifacts for deployment.

Feel free to extend the UI, customize the Dockerfile, or swap in a different backbone (e.g., `roberta-base`) via CLI flags.
