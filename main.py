# multilabel_emotion_transformer.py
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List
from inspect import signature

# --- Splits/metrics ---
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report, precision_recall_curve

# --- Torch / HF ---
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import transformers

# =========================
# Config
# =========================
SEED = 42
MODEL_NAME = "distilbert-base-uncased"   # try: "roberta-base", "bert-base-uncased", "microsoft/deberta-v3-base"
MAX_LENGTH = 256
EPOCHS = 4
TRAIN_BS = 16
EVAL_BS = 32
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

LABELS = ['admiration','amusement','gratitude','love','pride','relief','remorse']
TEXT_COL = 'text'
TRAIN_CSV_PATH = 'train.csv'     # training file
TEST_CSV_PATH  = 'dev.csv'       # held-out test file

np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# Utils: Version-safe TrainingArguments
# =========================
def build_training_args(**cfg):
    sig = signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())
    kwargs = dict(cfg)

    if "evaluation_strategy" not in allowed and "eval_strategy" in allowed:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy", "steps")
    elif "evaluation_strategy" not in allowed and "eval_strategy" not in allowed:
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

# =========================
# 1) Load data
# =========================
train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df  = pd.read_csv(TEST_CSV_PATH)

train_df[TEXT_COL] = train_df[TEXT_COL].fillna("").astype(str)
test_df[TEXT_COL]  = test_df[TEXT_COL].fillna("").astype(str)

y_train_full = train_df[LABELS].values.astype(int)
X_train_full = train_df[TEXT_COL].values

y_test = test_df[LABELS].values.astype(int)
X_test = test_df[TEXT_COL].values

# =========================
# 2) Stratified train/val split
# =========================
msss_inner = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
train_idx, val_idx = next(msss_inner.split(X_train_full, y_train_full))
X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# =========================
# 3) pos_weight for class imbalance
# =========================
pos_weight = []
N = y_train.shape[0]
for j in range(len(LABELS)):
    p = y_train[:, j].sum()
    w = (N - p) / p if p > 0 else 1.0
    pos_weight.append(float(np.clip(w, 1.0, 20.0)))
pos_weight = torch.tensor(pos_weight, dtype=torch.float)
print("pos_weight:", [float(x) for x in pos_weight])

# =========================
# 4) Tokenizer + Dataset (return lists, NOT tensors)
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray):
        self.texts = list(texts)
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding=False,         # let the collator pad
            max_length=MAX_LENGTH, # and control max length
        )
        # enc is a dict of lists: {'input_ids': [...], 'attention_mask': [...]}
        enc["labels"] = self.labels[idx].astype(np.float32)  # float32 for BCEWithLogits
        return enc

train_ds = SimpleDataset(X_train, y_train)
val_ds   = SimpleDataset(X_val,   y_val)
test_ds  = SimpleDataset(X_test,  y_test)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================
# 5) Model (multi-label) — NO monkey patch
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    problem_type="multi_label_classification",
)

# =========================
# 6) TrainingArguments + Trainer (version-safe) with custom loss
# =========================
training_args = build_training_args(
    output_dir="runs/emotion_clf",
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    logging_steps=100,
    eval_steps=300,
    save_steps=300,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_micro",
    warmup_ratio=WARMUP_RATIO,
    seed=SEED,
    report_to=[],
)

def provisional_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    micro = f1_score(labels, preds, average="micro", zero_division=0)
    macro = f1_score(labels, preds, average="macro", zero_division=0)
    return {"f1_micro": micro, "f1_macro": macro}

class PosWeightTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pull labels and move to device
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.float, device=logits.device)
        else:
            labels = labels.to(dtype=torch.float, device=logits.device)
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = PosWeightTrainer(
    model=model,
    args=training_args,
    train_dataset=SimpleDataset(X_train, y_train),
    eval_dataset=SimpleDataset(X_val, y_val),
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=provisional_metrics,
)

# =========================
# 7) Train
# =========================
trainer.train()

# =========================
# 8) Threshold tuning (per-label) on validation set
# =========================
model.eval()

def predict_proba(ds: torch.utils.data.Dataset, batch_size=EVAL_BS) -> np.ndarray:
    preds = []
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    device = model.device
    for batch in dl:
        batch_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            logits = model(**batch_inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)
    return np.vstack(preds)

print("Tuning thresholds on validation set...")
val_probs  = predict_proba(val_ds)
test_probs = predict_proba(test_ds)

def tune_thresholds(y_true: np.ndarray, probs: np.ndarray, labels: List[str]) -> Dict[str, float]:
    thresholds = {}
    for j, name in enumerate(labels):
        yj = y_true[:, j]
        pj = probs[:, j]
        if yj.sum() == 0:
            thresholds[name] = 0.5
            continue
        prec, rec, thr = precision_recall_curve(yj, pj)
        f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
        best_idx = f1s[:-1].argmax() if len(f1s) > 1 else 0
        best_thr = thr[best_idx] if len(thr) > 0 else 0.5
        thresholds[name] = float(best_thr)
    return thresholds

thresholds = tune_thresholds(y_val, val_probs, LABELS)
print("Tuned thresholds:", json.dumps(thresholds, indent=2))

def binarize_with_thresholds(probs: np.ndarray, thr_map: Dict[str, float]) -> np.ndarray:
    out = np.zeros_like(probs, dtype=int)
    for j, name in enumerate(LABELS):
        out[:, j] = (probs[:, j] >= thr_map[name]).astype(int)
    return out

# =========================
# 9) Final evaluation on test (dev.csv)
# =========================
y_pred_test = binarize_with_thresholds(test_probs, thresholds)
micro = f1_score(y_test, y_pred_test, average="micro", zero_division=0)
macro = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
print("\n--- FINAL TEST SET PERFORMANCE (dev.csv) ---")
print(f"Test Micro F1: {micro:.4f} | Test Macro F1: {macro:.4f}")
print("\nPer-label report:\n", classification_report(y_test, y_pred_test, target_names=LABELS, zero_division=0))

# =========================
# 10) Save artifacts
# =========================
print("Saving model and artifacts...")
os.makedirs("artifacts", exist_ok=True)
trainer.save_model("artifacts/model")
tokenizer.save_pretrained("artifacts/model")
with open("artifacts/thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)

# =========================
# 11) Inference helper
# =========================
def load_model_and_predict(texts: List[str]) -> pd.DataFrame:
    tok = AutoTokenizer.from_pretrained("artifacts/model")
    mdl = AutoModelForSequenceClassification.from_pretrained("artifacts/model")
    mdl.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)

    collator = DataCollatorWithPadding(tokenizer=tok)

    class _TmpDS(torch.utils.data.Dataset):
        def __init__(self, texts):
            self.texts = list(texts)
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            enc = tok(self.texts[idx], truncation=True, padding=False, max_length=MAX_LENGTH)
            return enc

    ds = _TmpDS(texts)
    preds = []
    dl = torch.utils.data.DataLoader(ds, batch_size=EVAL_BS, shuffle=False, collate_fn=collator)
    for batch in dl:
        batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}
        with torch.no_grad():
            logits = mdl(**batch).logits
            probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)
    probs = np.vstack(preds)

    with open("artifacts/thresholds.json", "r") as f:
        thr = json.load(f)
    binarized = binarize_with_thresholds(probs, thr)
    return pd.DataFrame(binarized, columns=LABELS)

# Example
if __name__ == "__main__":
    sample = ["Thanks for helping me yesterday! Really appreciate your kindness."]
    print("\nSample prediction:\n", load_model_and_predict(sample))