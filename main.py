import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from inspect import signature
from typing import Dict, List, Sequence

os.environ["TRANSFORMERS_NO_TF"] = "1"  # avoid TensorFlow deps when using PyTorch-only workflow

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from emotion_config import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_EVAL_PATH,
    DEFAULT_MAX_LEN,
    DEFAULT_MODEL,
    DEFAULT_TRAIN_PATH,
    LABELS,
    TEXT_COL,
)
from emotion_inference import EmotionPredictor


@dataclass
class TrainConfig:
    seed: int = 42
    model_name: str = DEFAULT_MODEL
    max_length: int = DEFAULT_MAX_LEN
    epochs: int = 4
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    eval_steps: int = 300
    logging_steps: int = 100
    val_size: float = 0.15
    train_path: str = DEFAULT_TRAIN_PATH
    eval_path: str = DEFAULT_EVAL_PATH
    output_dir: str = DEFAULT_ARTIFACT_DIR
    hf_output_dir: str = "runs/emotion_clf"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a multi-label emotion detector.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="HF model identifier.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN, help="Tokenizer max length.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--val-size", type=float, default=0.15, help="Fraction for validation split.")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--eval-path", default=DEFAULT_EVAL_PATH, help="Held-out dev/test CSV.")
    parser.add_argument("--output-dir", default="artifacts", help="Where to store artifacts.")
    parser.add_argument("--hf-output-dir", default="runs/emotion_clf", help="HF Trainer output.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return TrainConfig(
        seed=args.seed,
        model_name=args.model_name,
        max_length=args.max_length,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        val_size=args.val_size,
        train_path=args.train_path,
        eval_path=args.eval_path,
        output_dir=args.output_dir,
        hf_output_dir=args.hf_output_dir,
    )


# =========================
# Utility helpers
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return df


def stratified_split(texts: np.ndarray, labels: np.ndarray, test_size: float, seed: int):
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(texts, labels))
    return texts[train_idx], texts[val_idx], labels[train_idx], labels[val_idx]


class EmotionDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, texts: Sequence[str], labels: np.ndarray, max_length: int):
        self.tokenizer = tokenizer
        self.texts = list(texts)
        self.labels = labels.astype(np.float32)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )
        enc["labels"] = self.labels[idx]
        return enc


def compute_pos_weight(y: np.ndarray, max_weight: float = 20.0) -> torch.Tensor:
    weights = []
    n = y.shape[0]
    for j in range(y.shape[1]):
        positives = y[:, j].sum()
        if positives == 0:
            weights.append(1.0)
        else:
            weight = float(np.clip((n - positives) / positives, 1.0, max_weight))
            weights.append(weight)
    return torch.tensor(weights, dtype=torch.float)


def build_training_args(**cfg) -> TrainingArguments:
    """Wrapper that is tolerant to HF version differences."""
    sig = signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())
    kwargs = dict(cfg)

    if "evaluation_strategy" not in allowed and "eval_strategy" in allowed:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy", "steps")
    elif "evaluation_strategy" not in allowed:
        kwargs.pop("evaluation_strategy", None)
        if "evaluate_during_training" in allowed:
            kwargs["evaluate_during_training"] = True
        if "do_eval" in allowed:
            kwargs["do_eval"] = True

    if "save_strategy" not in allowed:
        kwargs.pop("save_strategy", None)
    if "load_best_model_at_end" not in allowed:
        kwargs.pop("load_best_model_at_end", None)
    if "metric_for_best_model" not in allowed:
        kwargs.pop("metric_for_best_model", None)
    if "warmup_ratio" not in allowed:
        kwargs.pop("warmup_ratio", None)

    final_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return TrainingArguments(**final_kwargs)


def provisional_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    micro = f1_score(labels, preds, average="micro", zero_division=0)
    macro = f1_score(labels, preds, average="macro", zero_division=0)
    return {"f1_micro": micro, "f1_macro": macro}


class PosWeightTrainer(Trainer):
    def __init__(self, pos_weight: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.float, device=logits.device)
        else:
            labels = labels.to(dtype=torch.float, device=logits.device)
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def predict_proba(model, dataset: Dataset, collator: DataCollatorWithPadding, batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    device = next(model.parameters()).device
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
    return np.vstack(preds)


def tune_thresholds(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    thresholds = {}
    for idx, label in enumerate(LABELS):
        y_label = y_true[:, idx]
        p_label = probs[:, idx]
        if y_label.sum() == 0:
            thresholds[label] = 0.5
            continue
        prec, rec, thr = precision_recall_curve(y_label, p_label)
        denom = prec + rec
        f1 = np.where(denom > 0, 2 * prec * rec / denom, 0)
        best_idx = f1[:-1].argmax() if len(f1) > 1 else 0
        thresholds[label] = float(thr[best_idx]) if len(thr) > 0 else 0.5
    return thresholds


def binarize_with_thresholds(probs: np.ndarray, thr_map: Dict[str, float]) -> np.ndarray:
    out = np.zeros_like(probs, dtype=int)
    for idx, label in enumerate(LABELS):
        out[:, idx] = (probs[:, idx] >= thr_map.get(label, 0.5)).astype(int)
    return out


def save_artifacts(cfg: TrainConfig, trainer: Trainer, tokenizer: AutoTokenizer, thresholds: Dict[str, float]) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    model_dir = os.path.join(cfg.output_dir, "model")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    with open(os.path.join(cfg.output_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def train_and_eval(cfg: TrainConfig) -> Dict[str, float]:
    set_seed(cfg.seed)
    train_df = read_frame(cfg.train_path)
    eval_df = read_frame(cfg.eval_path)

    y_train_full = train_df[LABELS].values.astype(int)
    X_train_full = train_df[TEXT_COL].values
    y_eval = eval_df[LABELS].values.astype(int)
    X_eval = eval_df[TEXT_COL].values

    X_train, X_val, y_train, y_val = stratified_split(X_train_full, y_train_full, cfg.val_size, cfg.seed)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Eval: {len(X_eval)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds = EmotionDataset(tokenizer, X_train, y_train, cfg.max_length)
    val_ds = EmotionDataset(tokenizer, X_val, y_val, cfg.max_length)
    eval_ds = EmotionDataset(tokenizer, X_eval, y_eval, cfg.max_length)

    pos_weight = compute_pos_weight(y_train)
    print("pos_weight:", [float(x) for x in pos_weight])

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = build_training_args(
        output_dir=cfg.hf_output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_micro",
        warmup_ratio=cfg.warmup_ratio,
        seed=cfg.seed,
        report_to=[],
    )

    trainer = PosWeightTrainer(
        pos_weight=pos_weight,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=provisional_metrics,
    )

    trainer.train()

    val_probs = predict_proba(trainer.model, val_ds, collator, cfg.eval_batch_size)
    eval_probs = predict_proba(trainer.model, eval_ds, collator, cfg.eval_batch_size)
    thresholds = tune_thresholds(y_val, val_probs)
    print("Tuned thresholds:", json.dumps(thresholds, indent=2))

    y_eval_pred = binarize_with_thresholds(eval_probs, thresholds)
    micro = f1_score(y_eval, y_eval_pred, average="micro", zero_division=0)
    macro = f1_score(y_eval, y_eval_pred, average="macro", zero_division=0)
    print("\n--- DEV SET PERFORMANCE ---")
    print(f"Micro F1: {micro:.4f} | Macro F1: {macro:.4f}")
    print("\nPer-label report:\n", classification_report(y_eval, y_eval_pred, target_names=LABELS, zero_division=0))

    save_artifacts(cfg, trainer, tokenizer, thresholds)
    return {"f1_micro": micro, "f1_macro": macro}


def main():
    cfg = parse_args()
    metrics = train_and_eval(cfg)
    print("\nSaved artifacts to:", cfg.output_dir)
    print("Metrics:", metrics)
    sample_text = ["Thanks for the reply! I appreciate your input. Please keep me in the loop, I'd love to be more active with this if possible."]
    predictor = EmotionPredictor(cfg.output_dir)
    sample_pred = predictor.predict_dataframe(sample_text)
    print("\nSample prediction:\n", pd.concat([pd.Series(sample_text, name=TEXT_COL), sample_pred], axis=1))


if __name__ == "__main__":
    main()
