# Emotion Explorer

A full workflow for multi-label emotion detection using Hugging Face transformers. The project includes:

- **Training pipeline** (`main.py`) that fine-tunes a sequence classifier on seven emotions (admiration, amusement, gratitude, love, pride, relief, remorse) using the provided `train.csv` / `dev.csv`.
- **Benchmark evaluation** (`benchmark.py`) for comprehensive model performance evaluation without requiring retraining.
- **Reusable inference module** (`emotion_inference.py`) plus shared config (`emotion_config.py`).
- **Notebook for Analysis** (`NLP_Assignment_Competitions.ipynb`): A comprehensive notebook for EDA and experimentation.
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
3. Benchmark the trained model (no retraining required):
    ```bash
    python benchmark.py
    ```
    This evaluates the already-trained model on test data and generates comprehensive performance metrics. Use `python benchmark.py --help` to see all options including custom model paths, output formats, and more.
4. Launch the web app locally (after training):
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
NLP_Assignment_Competitions.ipynb # notebook
benchmark.py           # comprehensive model evaluation without retraining
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

## Benchmark Evaluation

The `benchmark.py` script provides comprehensive evaluation of already-trained models without requiring retraining. This is ideal for:

- Evaluating model performance on test datasets
- Comparing different model versions
- Generating detailed performance reports
- Analyzing per-label performance metrics

### Usage Examples

```bash
# Basic benchmark with default settings (uses artifacts/model and dev.csv)
python benchmark.py

# Custom model path and test data
python benchmark.py --model-path /path/to/model --test-data /path/to/test.csv

# Output in different formats (json, csv, or both)
python benchmark.py --output-format both --output-dir my_results

# Adjust batch size and device for performance
python benchmark.py --batch-size 64 --device cuda

# Quiet mode for automated scripts
python benchmark.py --quiet --output-format csv
```

### Benchmark Output

The script generates several output files in the specified directory:

- `benchmark_metrics.json` - Comprehensive metrics in JSON format
- `benchmark_metrics.csv` - Flattened metrics in CSV format
- `predictions.csv` - Model predictions for each sample
- `probabilities.csv` - Probability scores for each label
- `classification_report.txt` - Detailed sklearn classification report

### Key Metrics

**Overall Performance:**
- Subset Accuracy (Exact Match)
- Hamming Loss
- Jaccard Score
- F1 scores (micro, macro, weighted, samples)
- Precision and Recall (micro, macro, weighted, samples)

**Per-Label Analysis:**
- Individual F1, precision, recall for each emotion
- Support counts (number of true instances)
- AUC-PR and AUC-ROC scores
- Label frequency statistics

### Command Line Options

```
--model-path        Path to trained model artifacts (default: artifacts)
--test-data         Path to test CSV file (default: dev.csv)
--output-dir        Directory for results (default: benchmark_results)
--output-format     Output format: json, csv, or both (default: json)
--batch-size        Batch size for inference (default: 32)
--device            Device: cpu, cuda, or auto (default: auto)
--no-save-predictions    Don't save individual predictions
--no-save-probabilities  Don't save probability scores
--quiet             Reduce verbose output
```

Feel free to extend the UI, customize the Dockerfile, swap in a different backbone (e.g., `roberta-base`) via CLI flags, or use the benchmark script for model comparison and performance analysis.
