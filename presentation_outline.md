# Emotion Explorer Presentation Outline

## 1) Problem & Goal
- Frame the task: multi-label emotion detection on short texts (admiration, amusement, gratitude, love, pride, relief, remorse).
- Business value: faster moderation, sentiment-aware analytics, feedback triage.
- Scope: English, single-sentence/short-paragraph inputs; not multi-turn conversations.

## 2) Dataset Overview
- Sources: `train.csv` (25,196 rows), `dev.csv` (3,149 rows); columns = `text` + 7 one-hot labels.
- Class balance: label counts — admiration 4,130; amusement 2,328; gratitude 2,662; love 2,086; pride 111; relief 153; remorse 545.
- Imbalance callout: pride/relief rare; motivates loss weighting and threshold tuning.
- Text length: median word count from EDA; distribution generally short (histogram screenshot if available).
- Data quality: missing text handling (fill empty with ""), stratified split to preserve label combos.

## 3) Modeling Choice
- Backbone: `distilbert-base-uncased` for speed/size trade-off (6 layers, ~66M params).
- Head: sigmoid multi-label classifier with 7 outputs.
- Why not larger? Keeps inference light for API/UI; can swap via `--model-name`.

## 4) Preprocessing Pipeline
- Tokenization: HF tokenizer, max length 256, truncation only (no padding until batch).
- Train/val split: multilabel stratified shuffle split with 15% validation to preserve co-occurrence patterns.
- Batching: dynamic padding via `DataCollatorWithPadding`.

## 5) Training Setup
- Loss: `BCEWithLogitsLoss` with per-label `pos_weight` to counter imbalance.
- Hyperparameters (artifacts/config.json): epochs 4; lr 2e-5; weight decay 0.01; warmup_ratio 0.06; batch sizes 16 train / 32 eval; eval/save every 300 steps; seed 42.
- Optimization: HF `Trainer` wrapper; no TensorFlow deps (`TRANSFORMERS_NO_TF=1`).
- Monitoring: provisional micro/macro F1 during training.

## 6) Threshold Tuning
- Method: per-label PR-curve sweep on validation set; choose threshold maximizing F1 per label.
- Motivation: single 0.5 cutoff underperforms on imbalanced labels.
- Resulting thresholds (examples): admiration 0.54; amusement 0.95; gratitude 0.96; love 0.70; pride 0.79; relief 0.91; remorse 0.86.

## 7) Evaluation (Dev Set, `benchmark.py`)
- Overall: micro-F1 0.843; macro-F1 0.754; subset accuracy 0.873; Hamming loss 0.022.
- Per-label F1: gratitude 0.925; remorse 0.891; amusement 0.849; love 0.839; admiration 0.800; pride 0.640; relief 0.333 (small support).
- Additional metrics: Jaccard 0.380; precision_micro 0.821; recall_micro 0.868; PR/ROC AUCs logged per label.
- Visuals: confusion-like co-occurrence heatmap; PR/ROC curves if generated; table or bar chart of per-label F1.

## 8) Error Analysis & Insights
- Failure modes: low data labels (pride, relief) under-detected; high thresholds reduce false positives but hurt recall.
- Common confusions: gratitude vs admiration; love vs admiration on supportive language.
- Qualitative examples: include 2–3 anonymized texts with predicted vs true labels to illustrate threshold impact.

## 9) Deployment & Usage
- Artifacts: `artifacts/model/` (weights + tokenizer), `artifacts/thresholds.json`, `artifacts/config.json`.
- Inference helper: `EmotionPredictor` (batching, thresholds applied) used by CLI and API.
- Serving: FastAPI + Jinja UI (`app/server.py`, `app/templates/index.html`); Dockerfile for containerized run.
- CLI snippets: `python main.py` to retrain; `python benchmark.py` to evaluate; `uvicorn app.server:app --reload` to serve locally.

## 10) Next Steps
- Data: collect/augment pride and relief examples; adversarial examples for robustness.
- Modeling: try `roberta-base` or domain-specific encoders; tune max_length; experiment with focal loss.
- Calibration: temperature scaling or Platt scaling; revisit thresholds after new data.
- Monitoring: track per-label drift and alert on precision/recall drops in production logs.

## 11) Appendix (for backup slides)
- Command line flags cheat sheet (`main.py --help`, `benchmark.py --help`).
- Hardware/runtime notes (GPU vs CPU throughput if measured).
- Link to repo structure and key files for reference.
