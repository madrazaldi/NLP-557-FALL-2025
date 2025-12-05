# Presentation Script – Emotion Explorer

## Slide 1: Problem & Goal
“We’re tackling multi-label emotion detection on short texts. Each message can carry multiple emotions—admiration, amusement, gratitude, love, pride, relief, remorse. The goal is to give downstream moderation, analytics, and feedback tools richer signals than a single sentiment score.”

## Slide 2: Data Snapshot
“Training set has 25,196 rows; dev set 3,149. Columns are `text` plus the 7 one-hot labels. Label counts show imbalance: admiration ~4.1k, amusement ~2.3k, gratitude ~2.7k, love ~2.1k, pride only 111, relief 153, remorse 545. Texts are short—median length is a few dozen words. Missing text is filled with empty strings.”

## Slide 3: Why This Model
“We chose `distilbert-base-uncased`: small and fast (6 layers, ~66M params) but strong for sentence-level tasks. On top, we add a sigmoid classifier head with seven outputs for multi-label predictions. We can swap in larger backbones via a flag if we need more capacity later.”

## Slide 4: Preprocessing
“Tokenization uses the HF tokenizer, max length 256, truncation only; padding happens dynamically per batch. We perform a multilabel stratified split—15% of the training data becomes validation to preserve label co-occurrence patterns. Dynamic padding uses the Hugging Face data collator.”

## Slide 5: Training Setup
“Loss is BCEWithLogits with per-label `pos_weight` to counter class imbalance. Hyperparameters: 4 epochs, learning rate 2e-5, weight decay 0.01, warmup ratio 0.06, batch sizes 16 train / 32 eval, eval and checkpoint every 300 steps, seed 42. Training runs through the HF Trainer with metrics reported as micro/macro F1 during training.”

## Slide 6: Threshold Tuning
“After training, we tune decision thresholds per label on the validation set. We sweep the precision–recall curve and pick the threshold that maximizes F1 for each label. This beats a single 0.5 cutoff on imbalanced labels. Example thresholds: admiration 0.54, amusement 0.95, gratitude 0.96, love 0.70, pride 0.79, relief 0.91, remorse 0.86.”

## Slide 7: Evaluation (Dev)
“On the dev set, micro-F1 is 0.843, macro-F1 0.754, subset accuracy 0.873, Hamming loss 0.022, Jaccard 0.380. Per-label F1: gratitude 0.925; remorse 0.891; amusement 0.849; love 0.839; admiration 0.800; pride 0.640; relief 0.333, where pride/relief are limited by low support. Precision_micro is 0.821 and recall_micro is 0.868.”

## Slide 8: Error Analysis
“Weak spots: pride and relief suffer from scarce data, leading to low recall. Confusions appear between gratitude vs admiration and love vs admiration on supportive language. Including 2–3 example texts helps show where the thresholds make or break the prediction.”

## Slide 9: Deployment
“Artifacts live in `artifacts/`: model weights and tokenizer under `model/`, thresholds JSON, and the training config snapshot. The `EmotionPredictor` applies the tuned thresholds and batching. Serving uses FastAPI with a Jinja UI; Dockerfile provided. Commands: `python main.py` to retrain, `python benchmark.py` to evaluate, `uvicorn app.server:app --reload` to run locally.”

## Slide 10: Next Steps
“Data: collect or augment pride and relief examples. Modeling: try `roberta-base` or focal loss. Calibration: temperature or Platt scaling and re-tune thresholds after new data. Monitoring: watch per-label precision/recall over time and alert on drift.”
