import json
import os
from typing import Dict, List, Sequence

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from emotion_config import DEFAULT_ARTIFACT_DIR, DEFAULT_MAX_LEN, LABELS


class _InferenceDataset(Dataset):
    def __init__(self, tokenizer, texts: Sequence[str], max_length: int):
        self.tokenizer = tokenizer
        self.texts = list(texts)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )


class EmotionPredictor:
    def __init__(self, artifact_dir: str = DEFAULT_ARTIFACT_DIR, device: str | None = None, batch_size: int = 32):
        self.artifact_dir = artifact_dir
        self.model_dir = os.path.join(self.artifact_dir, "model")
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"Model directory '{self.model_dir}' not found. Train the model first.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.thresholds = self._load_thresholds()
        self.max_length = self._read_max_length()
        self.batch_size = batch_size
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def _load_thresholds(self) -> Dict[str, float]:
        path = os.path.join(self.artifact_dir, "thresholds.json")
        with open(path, "r") as f:
            return json.load(f)

    def _read_max_length(self) -> int:
        cfg_path = os.path.join(self.artifact_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
                return int(cfg.get("max_length", DEFAULT_MAX_LEN))
        return DEFAULT_MAX_LEN

    def _predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        dataset = _InferenceDataset(self.tokenizer, texts, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(probs)
        return np.vstack(preds) if preds else np.zeros((0, len(LABELS)))

    def _binarize(self, probs: np.ndarray) -> np.ndarray:
        out = np.zeros_like(probs, dtype=int)
        for idx, label in enumerate(LABELS):
            thr = float(self.thresholds.get(label, 0.5))
            out[:, idx] = (probs[:, idx] >= thr).astype(int)
        return out

    def predict_dataframe(self, texts: Sequence[str]) -> pd.DataFrame:
        if len(texts) == 0:
            return pd.DataFrame(columns=LABELS)
        probs = self._predict_proba(texts)
        preds = self._binarize(probs)
        return pd.DataFrame(preds, columns=LABELS)

    def predict_with_probs(self, texts: Sequence[str]) -> Dict[str, List[Dict[str, float]]]:
        if len(texts) == 0:
            return {"predictions": [], "probabilities": []}
        probs = self._predict_proba(texts)
        preds = self._binarize(probs)
        prediction_dicts = []
        probability_dicts = []
        for row_pred, row_prob in zip(preds, probs):
            prediction_dicts.append({label: int(value) for label, value in zip(LABELS, row_pred)})
            probability_dicts.append({label: float(score) for label, score in zip(LABELS, row_prob)})
        return {"predictions": prediction_dicts, "probabilities": probability_dicts}
