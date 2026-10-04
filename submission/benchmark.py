import json
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

print("=" * 60)
print("LIGHTGBM CREDIT CARD FRAUD BENCHMARK")
print("=" * 60)

load_start = time.perf_counter()
df = pd.read_csv("creditcard.csv")
load_time = time.perf_counter() - load_start

X = df.drop(columns=["Class"])
y = df["Class"]

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

model = lgb.LGBMClassifier(objective="binary", n_estimators=1000, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1)

train_start = time.perf_counter()
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
training_time = time.perf_counter() - train_start

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

auc = roc_auc_score(y_test, probabilities)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, zero_division=0)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)

single_row = X_test.iloc[[0]]
model.predict_proba(single_row)

latencies = []

for _ in range(100):
    inference_start = time.perf_counter()
    model.predict_proba(single_row)
    latencies.append((time.perf_counter() - inference_start) * 1000)

latency_ms = float(np.median(latencies))

batch = X_test.iloc[:1000]
batch_start = time.perf_counter()
model.predict_proba(batch)
batch_time = time.perf_counter() - batch_start
throughput = len(batch) / batch_time

best_iteration = int(model.best_iteration_ or model.n_estimators_)

results = {
    "dataset_rows": int(df.shape[0]),
    "dataset_columns": int(df.shape[1]),
    "fraud_rows": int(y.sum()),
    "load_time_seconds": round(load_time, 6),
    "training_time_seconds": round(training_time, 6),
    "best_iteration": best_iteration,
    "auc_roc": round(float(auc), 6),
    "accuracy": round(float(accuracy), 6),
    "f1_score": round(float(f1), 6),
    "precision": round(float(precision), 6),
    "recall": round(float(recall), 6),
    "inference_latency_ms": round(latency_ms, 6),
    "inference_1000_rows_seconds": round(batch_time, 6),
    "inference_throughput_rows_per_second": round(throughput, 2),
    "python_version": platform.python_version(),
    "lightgbm_version": lgb.__version__
}

with open("benchmark_result.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print(f"Dataset shape       : {df.shape}")
print(f"Fraud rows          : {int(y.sum())}")
print(f"Load time           : {load_time:.6f} seconds")
print(f"Training time       : {training_time:.6f} seconds")
print(f"Best iteration      : {best_iteration}")
print(f"AUC-ROC             : {auc:.6f}")
print(f"Accuracy            : {accuracy:.6f}")
print(f"F1-score            : {f1:.6f}")
print(f"Precision           : {precision:.6f}")
print(f"Recall              : {recall:.6f}")
print(f"Latency (1 row)     : {latency_ms:.6f} ms")
print(f"Batch time (1000)   : {batch_time:.6f} seconds")
print(f"Throughput          : {throughput:.2f} rows/second")
print("Saved result        : benchmark_result.json")
print("=" * 60)
