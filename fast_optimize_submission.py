"""Fast entity-level blocking and Macro F0.5 benchmark for the challenge."""

import csv
import json
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

import run_full_pipeline as pipeline

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
BATCH_DIR = ARTIFACTS / "training_batches"
VALIDATION_DB = ARTIFACTS / "fast_validation_train_index_20260927.sqlite"
VALIDATION_DB = ARTIFACTS / "fast_validation_train_index_20260927_retry1.sqlite"
VALIDATION_SIZE = 20_000
BATCH_SIZE = 100
WORKERS = max(1, min(10, int((__import__("os").cpu_count() or 1) * 0.85)))
THRESHOLDS = np.array(
    [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
     0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98],
    dtype=float,
)


def load_training_entity_ids():
    training_ids = set()
    for path in sorted(BATCH_DIR.glob("batch_*.csv")):
        frame = pd.read_csv(path, usecols=["s1_id"], dtype=str)
        training_ids.update(frame["s1_id"])
    return training_ids


def sample_holdout_rows(training_ids):
    rng = random.Random(42)
    reservoir = []
    eligible = 0
    source_path = pipeline.TRAIN_DIR / "train_source1.tsv"
    with source_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        for record in reader:
            entity_id = record["entity_id"]
            if entity_id in training_ids:
                continue
            eligible += 1
            item = (
                entity_id,
                record["business_name"],
                record["business_address"],
                record["country"],
            )
            if len(reservoir) < VALIDATION_SIZE:
                reservoir.append(item)
            else:
                replacement = rng.randrange(eligible)
                if replacement < VALIDATION_SIZE:
                    reservoir[replacement] = item
    if len(reservoir) < VALIDATION_SIZE:
        raise RuntimeError(f"Only {len(reservoir)} eligible validation entities were found.")
    frame = pd.DataFrame(
        reservoir,
        columns=["entity_id", "business_name", "business_address", "country"],
    )
    pipeline.prepare_frame(frame)
    return pipeline.row_payloads(frame)


def build_validation_index():
    if VALIDATION_DB.exists():
        if not pipeline.training_index_is_complete(VALIDATION_DB):
            raise RuntimeError(
                f"Found an incomplete validation index at {VALIDATION_DB}; "
                "leaving it untouched. Choose a new index filename before retrying."
            )
        print(f"Reusing complete validation index: {VALIDATION_DB}", flush=True)
        return

    print("Building a separate disk-backed training S2/S3 index for validation...", flush=True)
    sources = [
        (pipeline.TRAIN_DIR / "train_source2.tsv", "S2"),
        (pipeline.TRAIN_DIR / "train_source3.tsv", "S3"),
    ]
    connection = pipeline.build_index(VALIDATION_DB, sources)
    connection.close()


def train_saved_batch_model():
    batches = sorted(BATCH_DIR.glob("batch_*.csv"))
    if not batches:
        raise FileNotFoundError(f"No saved training batches found in {BATCH_DIR}")
    first = pd.read_csv(batches[0])
    feature_columns = [
        column for column in first.columns if column not in {"s1_id", "candidate_id", "label"}
    ]
    scaler = StandardScaler()
    scaler.fit(first[feature_columns].to_numpy())
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=42)
    pair_count = 0
    for batch_path in batches:
        frame = pd.read_csv(batch_path)
        if frame.empty:
            continue
        model.partial_fit(
            scaler.transform(frame[feature_columns].to_numpy()),
            frame["label"].to_numpy(),
            classes=np.array([0, 1]),
        )
        pair_count += len(frame)
    print(f"Loaded existing model training: {pair_count:,} pairs; features={len(feature_columns)}", flush=True)
    return model, scaler, feature_columns


def score_validation_batch(rows):
    results = []
    for values in rows:
        row = pipeline.SimpleNamespace(**dict(zip(pipeline.ROW_FIELDS, values)))
        candidates = pipeline.candidates_for_row(pipeline._WORKER_CONNECTION, row)
        true_ids = pipeline.get_ground_truth(pipeline._WORKER_CONNECTION, row.entity_id)
        candidate_ids = set(candidates)
        covered_true = len(candidate_ids & true_ids)
        if candidates:
            features = np.asarray(
                [
                    pipeline.feature_vector(
                        row.name_norm,
                        row.address_norm,
                        row.country_norm,
                        row.address_numbers,
                        candidate,
                    )
                    for candidate in candidates.values()
                ],
                dtype=np.float32,
            )
            probabilities = pipeline._WORKER_MODEL.predict_proba(
                pipeline._WORKER_SCALER.transform(features)
            )[:, 1]
            labels = np.asarray(
                [entity_id in true_ids for entity_id in candidates], dtype=bool
            )
        else:
            probabilities = np.empty(0, dtype=float)
            labels = np.empty(0, dtype=bool)
        threshold_metrics = []
        for threshold in THRESHOLDS:
            predicted = probabilities >= threshold
            tp = int(np.count_nonzero(predicted & labels))
            fp = int(np.count_nonzero(predicted & ~labels))
            if not true_ids:
                f05 = 1.0 if fp == 0 else 0.0
                precision = 1.0 if fp == 0 else 0.0
                recall = 1.0
            else:
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / len(true_ids)
                f05 = pipeline.fbeta_05(precision, recall)
            threshold_metrics.append((f05, precision, recall, int(np.count_nonzero(predicted))))
        results.append((len(candidates), len(true_ids), covered_true, threshold_metrics))
    return results


def evaluate(rows, model, scaler):
    totals = np.zeros((len(THRESHOLDS), 4), dtype=np.float64)
    singleton_correct = np.zeros(len(THRESHOLDS), dtype=np.int64)
    empty_predictions = np.zeros(len(THRESHOLDS), dtype=np.int64)
    entity_counts = []
    total_true_links = 0
    covered_true_links = 0
    singleton_count = 0
    started = time.time()
    tasks = (rows[offset:offset + BATCH_SIZE] for offset in range(0, len(rows), BATCH_SIZE))
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=pipeline.initialize_worker,
        initargs=(str(VALIDATION_DB), model, scaler, None),
    ) as executor:
        for batch_result in pipeline.ordered_parallel_map(executor, score_validation_batch, tasks):
            for candidate_count, true_count, covered, threshold_metrics in batch_result:
                entity_counts.append(candidate_count)
                total_true_links += true_count
                covered_true_links += covered
                singleton_count += int(true_count == 0)
                totals[:, 0] += [metric[0] for metric in threshold_metrics]
                totals[:, 1] += [metric[1] for metric in threshold_metrics]
                totals[:, 2] += [metric[2] for metric in threshold_metrics]
                totals[:, 3] += [metric[3] for metric in threshold_metrics]
                empty_predictions += [metric[3] == 0 for metric in threshold_metrics]
                if true_count == 0:
                    singleton_correct += [metric[3] == 0 for metric in threshold_metrics]
    elapsed = time.time() - started
    scores = totals[:, 0] / len(rows)
    best_index = int(np.argmax(scores))
    best_threshold = float(THRESHOLDS[best_index])
    print("\nBEST ENTITY-LEVEL VALIDATION", flush=True)
    print(f"Entities: {len(rows):,}; training-ID overlap: 0; runtime: {elapsed:.1f}s", flush=True)
    print(f"Candidate recall: {covered_true_links / total_true_links:.2%} ({covered_true_links:,}/{total_true_links:,})", flush=True)
    print(f"Average candidates: {np.mean(entity_counts):.1f}", flush=True)
    print(f"95th percentile candidates: {np.percentile(entity_counts, 95):.0f}", flush=True)
    print(f"Maximum candidates: {max(entity_counts):,}", flush=True)
    print(f"Macro F0.5: {scores[best_index]:.2%}", flush=True)
    print(f"Macro precision: {totals[best_index, 1] / len(rows):.2%}", flush=True)
    print(f"Macro recall: {totals[best_index, 2] / len(rows):.2%}", flush=True)
    singleton_correct = (len(rows) - singleton_count) + (totals[best_index, 3] == 0)
    singleton_accuracy = singleton_correct[best_index] / singleton_count if singleton_count else 0.0
    print(f"Singleton accuracy: {singleton_accuracy:.2%} ({singleton_count:,} true singletons)", flush=True)
    print(f"Predicted matches: {int(totals[best_index, 3]):,}", flush=True)
    print(f"Empty predictions: {int(empty_predictions[best_index]):,}", flush=True)
    print(f"Threshold: {best_threshold:.2f}", flush=True)
    print(f"Inference runtime: {elapsed:.1f}s", flush=True)

    model_path = ARTIFACTS / "fast_validation_model.pkl"
    model_path = ARTIFACTS / "fast_validation_model_20260927.pkl"
    config_path = ARTIFACTS / "fast_validation_config_20260927.json"
    with model_path.open("wb") as output:
        pickle.dump({"model": model, "scaler": scaler}, output, protocol=pickle.HIGHEST_PROTOCOL)
    config = {
        "threshold": best_threshold,
        "macro_f05": float(scores[best_index]),
        "candidate_recall": covered_true_links / total_true_links if total_true_links else 0.0,
        "average_candidates": float(np.mean(entity_counts)),
        "p95_candidates": float(np.percentile(entity_counts, 95)),
        "maximum_candidates": int(max(entity_counts)),
        "workers": WORKERS,
        "feature_columns": pipeline.FEATURE_NAMES,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved benchmark model/config: {model_path}; {config_path}", flush=True)
    return scores[best_index], best_threshold


def main():
    started = time.time()
    training_ids = load_training_entity_ids()
    print(f"Saved-batch training entities: {len(training_ids):,}", flush=True)
    rows = sample_holdout_rows(training_ids)
    print(f"Sampled {len(rows):,} entity-disjoint validation S1 rows.", flush=True)
    build_validation_index()
    model, scaler, _ = train_saved_batch_model()
    score, threshold = evaluate(rows, model, scaler)
    print(
        f"\nFast benchmark complete in {time.time() - started:.1f}s; "
        f"best macro F0.5={score:.4f} at threshold={threshold:.2f}.",
        flush=True,
    )
    if score < 0.80:
        print("Target 0.80 not reached; do not treat the candidate-conditioned batch score as a leaderboard estimate.", flush=True)


if __name__ == "__main__":
    main()
