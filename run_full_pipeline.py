# -*- coding: utf-8 -*-
"""Train on the full labeled data and generate a complete submission in batches."""

import csv
import gc
import json
import os
import pickle
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import unicodedata
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import partial_ratio, ratio, token_set_ratio
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RESOURCE = ROOT / "student_resource"
TRAIN_DIR = RESOURCE / "dataset" / "train"
TEST_DIR = RESOURCE / "dataset" / "test"
OUTPUT_DIR = RESOURCE / "output"
ROOT_OUTPUT_DIR = ROOT / "output"
ARTIFACTS_DIR = ROOT / "artifacts"
DB_PATH = ARTIFACTS_DIR / "full_pipeline_index.sqlite"

SOURCE_CHUNK_SIZE = 20_000
S1_BATCH_SIZE = 100
MAX_CANDIDATES_PER_SOURCE = 1000
MAX_CANDIDATES_PER_SOURCE = 1000
POSTINGS_PER_TOKEN_LIMIT = 500
NEGATIVE_MULTIPLIER = 10
VALIDATION_MODULUS = 100
RANDOM_STATE = 42
THRESHOLDS = np.arange(0.10, 0.91, 0.05)
# Use ten workers while keeping the in-flight batch window bounded.
# Use approximately 85% of logical CPUs, capped at ten for memory safety.
PARALLEL_WORKERS = max(1, min(10, int((os.cpu_count() or 1) * 0.85)))

DEVANAGARI_MAP = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "अं": "am", "अः": "ah",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ं": "n", "ँ": "n",
    "्": "", "़": "", "।": ".", "॥": ".",
}

NAME_ABBREVIATIONS = {
    "pvt": "private", "ltd": "limited", "corp": "corporation", "inc": "incorporated",
    "llc": "limited liability company", "co": "company", "intl": "international",
    "mfg": "manufacturing", "tech": "technologies", "dept": "department",
    "univ": "university", "assoc": "association", "grp": "group",
    "svc": "services", "svcs": "services", "ent": "enterprises",
    "mgmt": "management", "soln": "solutions", "solns": "solutions",
    "sarl": "societe a responsabilite limitee",
    "sasu": "societe par actions simplifiee unipersonnelle",
    "sas": "societe par actions simplifiee",
    "eurl": "entreprise unipersonnelle a responsabilite limitee",
    "sci": "societe civile immobiliere", "sa": "societe anonyme",
    "ei": "entreprise individuelle", "ets": "etablissements", "etbl": "etablissements",
    "cie": "compagnie", "ste": "societe",
}

ADDRESS_ABBREVIATIONS = {
    "rd": "road", "st": "street", "str": "strasse", "ave": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "sq": "square", "pkwy": "parkway", "hwy": "highway",
    "fl": "floor", "ste": "suite", "apt": "apartment", "bldg": "building",
    "nr": "near", "opp": "opposite", "pk": "park", "bd": "boulevard",
    "bvd": "boulevard", "av": "avenue", "r": "rue", "imp": "impasse",
    "all": "allee", "rte": "route", "bat": "batiment", "res": "residence",
    "etg": "etage", "marg": "road", "rasta": "road", "chowk": "square",
    "bazar": "market", "bazaar": "market", "stn": "station", "sec": "sector",
    "soc": "society", "col": "colony", "apts": "apartments",
}

LEGAL_STOPWORDS = {
    "incorporated", "corporation", "limited", "private", "company", "technologies",
    "services", "enterprises", "international", "group", "solutions", "management",
    "llc", "inc", "ltd", "pvt", "corp", "co", "societe", "actions",
    "responsabilite", "limitee", "simplifiee", "unipersonnelle", "entreprise",
    "etablissements", "compagnie", "sarl", "sasu", "sas", "eurl", "sci", "sa", "ei", "ets",
}

NAME_ABBR_REGEX = re.compile(r"\b(" + "|".join(map(re.escape, NAME_ABBREVIATIONS)) + r")\b")
ADDRESS_ABBR_REGEX = re.compile(r"\b(" + "|".join(map(re.escape, ADDRESS_ABBREVIATIONS)) + r")\b")
FEATURE_NAMES = (
    "name_ratio", "name_token_set_ratio", "name_partial_ratio",
    "address_ratio", "address_token_set_ratio", "address_partial_ratio",
    "country_match", "num_overlap_count", "name_len_s1", "name_len_candidate",
    "address_len_s1", "address_len_candidate",
)
ROW_FIELDS = (
    "entity_id", "name_norm", "address_norm", "country_norm", "address_numbers"
)
_WORKER_CONNECTION = None
_WORKER_MODEL = None
_WORKER_SCALER = None
_WORKER_THRESHOLD = None


def initialize_worker(db_path, model=None, scaler=None, threshold=None):
    global _WORKER_CONNECTION, _WORKER_MODEL, _WORKER_SCALER, _WORKER_THRESHOLD
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    _WORKER_CONNECTION = sqlite3.connect(uri, uri=True, timeout=60)
    _WORKER_CONNECTION.execute("PRAGMA query_only=ON")
    _WORKER_CONNECTION.execute("PRAGMA cache_size=-32768")
    _WORKER_CONNECTION.execute("PRAGMA mmap_size=134217728")
    _WORKER_MODEL = model
    _WORKER_SCALER = scaler
    _WORKER_THRESHOLD = threshold


def ordered_parallel_map(executor, worker, tasks):
    """Keep only a small, ordered window of batches in flight."""
    task_iterator = iter(tasks)
    pending = deque()
    window = PARALLEL_WORKERS
    for _ in range(window):
        try:
            pending.append(executor.submit(worker, next(task_iterator)))
        except StopIteration:
            break
    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            pending.append(executor.submit(worker, next(task_iterator)))
        except StopIteration:
            pass


def row_payloads(frame):
    return [
        (str(row.entity_id), row.name_norm, row.address_norm, row.country_norm, row.address_numbers)
        for row in frame.itertuples(index=False)
    ]


def training_batch_worker(payload):
    batch_number, rows, sample_negatives, include_positives = payload
    rng = random.Random(RANDOM_STATE + batch_number)
    features = []
    labels = []
    group_sizes = []
    truth_counts = []
    for values in rows:
        row = SimpleNamespace(**dict(zip(ROW_FIELDS, values)))
        true_ids = get_ground_truth(_WORKER_CONNECTION, row.entity_id)
        candidates = candidates_for_row(
            _WORKER_CONNECTION,
            row,
            include_positives=include_positives,
            ground_truth=true_ids,
        )
        positives = [candidate for entity_id, candidate in candidates.items() if entity_id in true_ids]
        negatives = [candidate for entity_id, candidate in candidates.items() if entity_id not in true_ids]
        if sample_negatives:
            max_negatives = max(10, len(positives) * NEGATIVE_MULTIPLIER)
            if len(negatives) > max_negatives:
                negatives = rng.sample(negatives, max_negatives)
        group_sizes.append(len(positives) + len(negatives))
        truth_counts.append(len(true_ids))
        for candidate in positives:
            features.append(feature_vector(
                row.name_norm, row.address_norm, row.country_norm, row.address_numbers, candidate
            ))
            labels.append(1)
        for candidate in negatives:
            features.append(feature_vector(
                row.name_norm, row.address_norm, row.country_norm, row.address_numbers, candidate
            ))
            labels.append(0)
    if not features:
        return (
            np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty(0, dtype=np.int8),
            len(rows),
            group_sizes,
            truth_counts,
        )
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        len(rows),
        group_sizes,
        truth_counts,
    )


def inference_batch_worker(rows):
    results = []
    for values in rows:
        row = SimpleNamespace(**dict(zip(ROW_FIELDS, values)))
        candidates = candidates_for_row(_WORKER_CONNECTION, row)
        candidate_ids = sorted(candidates)
        if candidates:
            matrix = np.asarray([
                feature_vector(
                    row.name_norm, row.address_norm, row.country_norm,
                    row.address_numbers, candidate,
                )
                for candidate in candidates.values()
            ], dtype=np.float32)
            probabilities = _WORKER_MODEL.predict_proba(_WORKER_SCALER.transform(matrix))[:, 1]
            matched_ids = sorted(
                entity_id for entity_id, probability in zip(candidates, probabilities)
                if probability >= _WORKER_THRESHOLD
            )
        else:
            matched_ids = []
        results.append((row.entity_id, ",".join(matched_ids), ",".join(candidate_ids)))
    return results


def transliterate_devanagari(value):
    text = str(value or "")
    return "".join(DEVANAGARI_MAP.get(char, char) for char in text)


def normalize_series(series, abbreviations, abbreviation_regex):
    values = series.fillna("").astype(str).str.lower().apply(transliterate_devanagari)
    values = (
        values.str.normalize("NFKD")
        .str.replace(r"[\u0300-\u036f]", "", regex=True)
        .str.replace("\ufffd", " ", regex=False)
        .str.replace("&", " and ", regex=False)
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    values = values.apply(lambda text: abbreviation_regex.sub(lambda match: abbreviations[match.group(0)], text))
    return values.str.replace(r"\s+", " ", regex=True).str.strip()


def name_tokens(text):
    tokens = [token for token in str(text).split() if len(token) >= 3]
    distinctive = [token for token in tokens if token not in LEGAL_STOPWORDS]
    return distinctive or tokens


def address_numbers(text):
    return re.findall(r"\d+", str(text or ""))


def prepare_frame(frame):
    frame["name_norm"] = normalize_series(frame["business_name"], NAME_ABBREVIATIONS, NAME_ABBR_REGEX)
    frame["address_norm"] = normalize_series(
        frame["business_address"], ADDRESS_ABBREVIATIONS, ADDRESS_ABBR_REGEX
    )
    frame["country_norm"] = frame["country"].fillna("").astype(str).str.lower().str.strip()
    frame["address_numbers"] = frame["business_address"].fillna("").astype(str).apply(address_numbers)
    return frame


def connect_db(path):
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-100000")
    connection.executescript(
        """
        CREATE TABLE records (
            source TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            name_norm TEXT NOT NULL,
            address_norm TEXT NOT NULL,
            country_norm TEXT NOT NULL,
            address_numbers TEXT NOT NULL,
            PRIMARY KEY (source, entity_id)
        ) WITHOUT ROWID;
        CREATE TABLE postings (
            source TEXT NOT NULL,
            country_norm TEXT NOT NULL,
            token TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            PRIMARY KEY (source, country_norm, token, entity_id)
        ) WITHOUT ROWID;
        CREATE TABLE ground_truth (
            s1_id TEXT PRIMARY KEY,
            candidate_ids TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    return connection


def build_record_index(connection, path, source):
    print(f"Indexing {path.name} ({source}) ...", flush=True)
    started = time.time()
    record_sql = "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)"
    posting_sql = "INSERT OR IGNORE INTO postings VALUES (?, ?, ?, ?)"
    total_rows = 0
    for chunk in pd.read_csv(path, sep="\t", chunksize=SOURCE_CHUNK_SIZE):
        prepare_frame(chunk)
        records = []
        postings = []
        for row in chunk.itertuples(index=False):
            entity_id = str(row.entity_id)
            country = row.country_norm
            records.append((
                source, entity_id, row.name_norm, row.address_norm, country,
                ",".join(row.address_numbers),
            ))
            postings.extend(
                (source, country, "N:" + token, entity_id)
                for token in set(name_tokens(row.name_norm))
            )
            postings.extend(
                (source, country, "A:" + number, entity_id)
                for number in set(row.address_numbers)
            )
        connection.executemany(record_sql, records)
        connection.executemany(posting_sql, postings)
        connection.commit()
        total_rows += len(chunk)
        print(f"  indexed {total_rows:,} rows in {time.time() - started:.1f}s", flush=True)
    print(f"  completed {source}: {total_rows:,} rows", flush=True)


def build_ground_truth(connection):
    path = TRAIN_DIR / "train_ground_truth.tsv"
    print("Indexing ground truth ...", flush=True)
    total_rows = 0
    for chunk in pd.read_csv(path, sep="\t", chunksize=SOURCE_CHUNK_SIZE):
        entries = []
        for row in chunk.itertuples(index=False):
            candidates = "" if pd.isna(row.matched_entity_ids) else str(row.matched_entity_ids).strip()
            entries.append((str(row.source1_entity_id), candidates))
        connection.executemany("INSERT INTO ground_truth VALUES (?, ?)", entries)
        connection.commit()
        total_rows += len(chunk)
        print(f"  loaded {total_rows:,} labels", flush=True)


def get_ground_truth(connection, s1_id):
    row = connection.execute(
        "SELECT candidate_ids FROM ground_truth WHERE s1_id = ?", (s1_id,)
    ).fetchone()
    if not row or not row[0]:
        return set()
    return {candidate.strip() for candidate in row[0].split(",") if candidate.strip()}


def get_record_candidates(connection, source, country, tokens, limit):
    if not tokens:
        return []
    token_counts = {}
    for token in sorted(set(tokens)):
        rows = connection.execute(
            "SELECT entity_id FROM postings "
            "WHERE source = ? AND country_norm = ? AND token = ? LIMIT ?",
            (source, country, token, POSTINGS_PER_TOKEN_LIMIT),
        )
        for (entity_id,) in rows:
            token_counts[entity_id] = token_counts.get(entity_id, 0) + 1

    ranked_ids = sorted(token_counts, key=lambda entity_id: (-token_counts[entity_id], entity_id))[:limit]
    records = get_records_by_ids(connection, source, ranked_ids)
    records_by_id = {record[0]: record for record in records}
    return [records_by_id[entity_id] for entity_id in ranked_ids if entity_id in records_by_id]


def get_records_by_ids(connection, source, entity_ids):
    if not entity_ids:
        return []
    placeholders = ",".join("?" for _ in entity_ids)
    query = f"""
        SELECT entity_id, name_norm, address_norm, country_norm, address_numbers
        FROM records WHERE source = ? AND entity_id IN ({placeholders})
    """
    return connection.execute(query, (source, *entity_ids)).fetchall()


def feature_vector(s1_name, s1_address, s1_country, s1_numbers, candidate):
    _, candidate_name, candidate_address, candidate_country, candidate_numbers = candidate
    candidate_numbers = candidate_numbers.split(",") if candidate_numbers else []
    return [
        ratio(s1_name, candidate_name) / 100.0,
        token_set_ratio(s1_name, candidate_name) / 100.0,
        partial_ratio(s1_name, candidate_name) / 100.0,
        ratio(s1_address, candidate_address) / 100.0,
        token_set_ratio(s1_address, candidate_address) / 100.0,
        partial_ratio(s1_address, candidate_address) / 100.0,
        float(s1_country == candidate_country),
        float(len(set(s1_numbers) & set(candidate_numbers))),
        float(len(s1_name)), float(len(candidate_name)),
        float(len(s1_address)), float(len(candidate_address)),
    ]


def candidates_for_row(connection, row, include_positives=False, ground_truth=None):
    blocking_tokens = ["N:" + token for token in set(name_tokens(row.name_norm))]
    blocking_tokens.extend("A:" + number for number in set(row.address_numbers))
    candidates = {}
    for source in ("S2", "S3"):
        source_records = get_record_candidates(
            connection, source, row.country_norm, blocking_tokens, MAX_CANDIDATES_PER_SOURCE
        )
        candidates.update({candidate[0]: candidate for candidate in source_records})

    if include_positives and ground_truth:
        missing_s2 = [candidate for candidate in ground_truth if candidate.startswith("S2-") and candidate not in candidates]
        missing_s3 = [candidate for candidate in ground_truth if candidate.startswith("S3-") and candidate not in candidates]
        for source, entity_ids in (("S2", missing_s2), ("S3", missing_s3)):
            candidates.update({candidate[0]: candidate for candidate in get_records_by_ids(connection, source, entity_ids)})

    return candidates


def labeled_features(connection, row, rng, sample_negatives=True):
    true_ids = get_ground_truth(connection, row.entity_id)
    candidates = candidates_for_row(connection, row, include_positives=True, ground_truth=true_ids)
    positives = [candidate for entity_id, candidate in candidates.items() if entity_id in true_ids]
    negatives = [candidate for entity_id, candidate in candidates.items() if entity_id not in true_ids]
    if sample_negatives:
        max_negatives = max(10, len(positives) * NEGATIVE_MULTIPLIER)
        if len(negatives) > max_negatives:
            negatives = rng.sample(negatives, max_negatives)
    pairs = [(candidate, 1) for candidate in positives]
    pairs.extend((candidate, 0) for candidate in negatives)
    return [
        (feature_vector(row.name_norm, row.address_norm, row.country_norm, row.address_numbers, candidate), label)
        for candidate, label in pairs
    ]


def is_validation_id(entity_id):
    suffix = entity_id.rsplit("-", 1)[-1]
    return suffix.isdigit() and int(suffix) % VALIDATION_MODULUS == 0


def fbeta_05(precision, recall):
    denominator = 0.25 * precision + recall
    return (1.25 * precision * recall) / denominator if denominator else 0.0


def fit_and_select_threshold(db_path):
    source1_path = TRAIN_DIR / "train_source1.tsv"
    checkpoint_path = ARTIFACTS_DIR / "full_training_checkpoint.pkl"
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    scaler_fitted = False
    trained_pairs = 0
    trained_entities = 0
    resume_batch_number = 0
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)
        model = checkpoint["model"]
        scaler = checkpoint["scaler"]
        scaler_fitted = checkpoint["scaler_fitted"]
        trained_pairs = checkpoint["trained_pairs"]
        trained_entities = checkpoint["trained_entities"]
        resume_batch_number = checkpoint["next_batch_number"]
        print(
            f"Resuming training after batch {resume_batch_number:,}: "
            f"{trained_entities:,} entities, {trained_pairs:,} pairs already processed.",
            flush=True,
        )
    started = time.time()
    print(
        f"Training with {PARALLEL_WORKERS} workers over all non-holdout training rows ...",
        flush=True,
    )

    def make_training_tasks():
        for batch_number, chunk in enumerate(
            pd.read_csv(source1_path, sep="\t", chunksize=S1_BATCH_SIZE)
        ):
            if batch_number < resume_batch_number:
                continue
            prepare_frame(chunk)
            rows = [row for row in row_payloads(chunk) if not is_validation_id(row[0])]
            yield batch_number, rows, True, True

    def make_validation_tasks():
        for batch_number, chunk in enumerate(
            pd.read_csv(source1_path, sep="\t", chunksize=S1_BATCH_SIZE)
        ):
            prepare_frame(chunk)
            rows = [row for row in row_payloads(chunk) if is_validation_id(row[0])]
            if rows:
                yield batch_number, rows, False, False

    with ProcessPoolExecutor(
        max_workers=PARALLEL_WORKERS,
        initializer=initialize_worker,
        initargs=(str(db_path),),
    ) as executor:
        for batch_number, (matrix, target, batch_entities, _, _) in enumerate(
            ordered_parallel_map(executor, training_batch_worker, make_training_tasks()),
            start=resume_batch_number + 1,
        ):
            if len(target):
                if not scaler_fitted:
                    scaler.fit(matrix)
                    scaler_fitted = True
                model.partial_fit(scaler.transform(matrix), target, classes=np.array([0, 1]))
                trained_pairs += len(target)
            trained_entities += batch_entities
            if batch_number % 100 == 0:
                checkpoint = {
                    "model": model,
                    "scaler": scaler,
                    "scaler_fitted": scaler_fitted,
                    "trained_pairs": trained_pairs,
                    "trained_entities": trained_entities,
                    "next_batch_number": batch_number + 1,
                }
                temporary_checkpoint = checkpoint_path.with_suffix(".pkl.tmp")
                with temporary_checkpoint.open("wb") as checkpoint_file:
                    pickle.dump(checkpoint, checkpoint_file, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temporary_checkpoint, checkpoint_path)
            if batch_number % 20 == 0:
                print(
                    f"  train batch {batch_number:,} | entities {trained_entities:,} | "
                    f"pairs {trained_pairs:,} | elapsed {time.time() - started:.1f}s",
                    flush=True,
                )
            del matrix, target

        if not scaler_fitted:
            raise RuntimeError("No training pairs were generated; cannot fit the model.")

        validation_scores = [[] for _ in THRESHOLDS]
        validation_pairs = 0
        validation_entities = 0
        true_links = 0
        retrieved_true_links = 0
        print("Scoring held-out training entities with parallel workers ...", flush=True)
        for matrix, labels, batch_entities, group_sizes, truth_counts in ordered_parallel_map(
            executor, training_batch_worker, make_validation_tasks()
        ):
            if len(labels):
                probabilities = model.predict_proba(scaler.transform(matrix))[:, 1]
            else:
                probabilities = np.empty(0, dtype=np.float64)
            offset = 0
            for group_size, truth_count in zip(group_sizes, truth_counts):
                group_probabilities = probabilities[offset:offset + group_size]
                group_labels = labels[offset:offset + group_size]
                offset += group_size
                true_links += truth_count
                retrieved_true_links += int(np.count_nonzero(group_labels == 1))
                for index, threshold in enumerate(THRESHOLDS):
                    predicted = group_probabilities >= threshold
                    tp = int(np.count_nonzero(predicted & (group_labels == 1)))
                    fp = int(np.count_nonzero(predicted & (group_labels == 0)))
                    if truth_count == 0:
                        validation_scores[index].append(1.0 if fp == 0 else 0.0)
                    else:
                        fn = truth_count - tp
                        precision = tp / (tp + fp) if tp + fp else 0.0
                        recall = tp / truth_count
                        validation_scores[index].append(fbeta_05(precision, recall))
                validation_pairs += len(labels)
            validation_entities += batch_entities
            del matrix, labels
            gc.collect()

    scores = [
        (float(np.mean(entity_scores)), float(threshold))
        for entity_scores, threshold in zip(validation_scores, THRESHOLDS)
        if entity_scores
    ]
    best_score, best_threshold = max(scores)
    print(
        f"Training done: {trained_entities:,} entities, {trained_pairs:,} pairs; "
        f"validation {validation_entities:,} entities / {validation_pairs:,} pairs; "
        f"macro F0.5={best_score:.4f}, threshold={best_threshold:.2f}; "
        f"candidate recall={retrieved_true_links / true_links:.4f}",
        flush=True,
    )
    return model, scaler, best_threshold, best_score


def fit_validation_entities(db_path, model, scaler):
    source1_path = TRAIN_DIR / "train_source1.tsv"
    trained_entities = 0
    trained_pairs = 0
    started = time.time()

    def make_validation_training_tasks():
        for batch_number, chunk in enumerate(
            pd.read_csv(source1_path, sep="\t", chunksize=S1_BATCH_SIZE)
        ):
            prepare_frame(chunk)
            rows = [row for row in row_payloads(chunk) if is_validation_id(row[0])]
            if not rows:
                continue
            yield batch_number, rows, True, True

    print("Adding held-out labels after threshold selection so the final model sees all training entities ...", flush=True)
    with ProcessPoolExecutor(
        max_workers=PARALLEL_WORKERS,
        initializer=initialize_worker,
        initargs=(str(db_path),),
    ) as executor:
        for batch_number, (matrix, target, batch_entities, _, _) in enumerate(
            ordered_parallel_map(executor, training_batch_worker, make_validation_training_tasks()), start=1
        ):
            if len(target):
                model.partial_fit(scaler.transform(matrix), target, classes=np.array([0, 1]))
                trained_pairs += len(target)
            trained_entities += batch_entities
            if batch_number % 20 == 0:
                print(
                    f"  held-out training batch {batch_number:,} | entities {trained_entities:,} | "
                    f"pairs {trained_pairs:,} | elapsed {time.time() - started:.1f}s",
                    flush=True,
                )
            del matrix, target
            gc.collect()

    if not trained_pairs:
        raise RuntimeError("No labeled pairs were generated for held-out final training.")
    print(
        f"Held-out final training complete: {trained_entities:,} entities, "
        f"{trained_pairs:,} pairs in {time.time() - started:.1f}s",
        flush=True,
    )
    return model


def build_index(path, sources):
    connection = connect_db(path)
    try:
        for input_path, source in sources:
            build_record_index(connection, input_path, source)
        if any(source in {"S2", "S3"} for _, source in sources) and sources[0][0].parent == TRAIN_DIR:
            build_ground_truth(connection)
    except Exception:
        connection.close()
        raise
    return connection


def training_index_is_complete(path):
    if not path.is_file():
        return False
    uri = f"file:{path.as_posix()}?mode=ro"
    connection = None
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=10)
        source_counts = dict(connection.execute(
            "SELECT source, COUNT(*) FROM records GROUP BY source"
        ).fetchall())
        label_count = connection.execute("SELECT COUNT(*) FROM ground_truth").fetchone()[0]
        return (
            source_counts.get("S2") == 5_034_616
            and source_counts.get("S3") == 5_285_603
            and label_count == 2_206_821
        )
    except sqlite3.Error:
        return False
    finally:
        if connection is not None:
            connection.close()


def write_test_outputs(db_path, model, scaler, threshold):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matching_path = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"
    matching_tmp = matching_path.with_suffix(".tsv.tmp")
    candidate_tmp = candidate_path.with_suffix(".tsv.tmp")
    matching_tmp.unlink(missing_ok=True)
    candidate_tmp.unlink(missing_ok=True)

    source1_path = TEST_DIR / "test_source1.tsv"
    total_entities = 0
    started = time.time()
    with matching_tmp.open("w", encoding="utf-8", newline="") as matching_file, candidate_tmp.open(
        "w", encoding="utf-8", newline=""
    ) as candidate_file:
        matching_writer = csv.writer(matching_file, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
        matching_writer.writerow(("source1_entity_id", "matched_entity_ids"))
        candidate_writer.writerow(("source1_entity_id", "candidate_entity_ids"))

        def make_inference_tasks():
            for chunk in pd.read_csv(source1_path, sep="\t", chunksize=S1_BATCH_SIZE):
                prepare_frame(chunk)
                yield row_payloads(chunk)

        with ProcessPoolExecutor(
            max_workers=PARALLEL_WORKERS,
            initializer=initialize_worker,
            initargs=(str(db_path), model, scaler, threshold),
        ) as executor:
            for batch_results in ordered_parallel_map(
                executor, inference_batch_worker, make_inference_tasks()
            ):
                for entity_id, matched_ids, candidate_ids in batch_results:
                    matching_writer.writerow((entity_id, matched_ids))
                    candidate_writer.writerow((entity_id, candidate_ids))
                total_entities += len(batch_results)
                matching_file.flush()
                candidate_file.flush()
                if total_entities % 10_000 == 0:
                    print(
                        f"  test inference {total_entities:,} / 1,732,544 S1 rows; "
                        f"elapsed {time.time() - started:.1f}s",
                        flush=True,
                    )

    os.replace(matching_tmp, matching_path)
    os.replace(candidate_tmp, candidate_path)
    shutil.copy2(matching_path, ROOT_OUTPUT_DIR / matching_path.name)
    shutil.copy2(candidate_path, ROOT_OUTPUT_DIR / candidate_path.name)
    print(f"Wrote {total_entities:,} rows to {matching_path} and {candidate_path}", flush=True)


def main():
    started = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if training_index_is_complete(DB_PATH):
        print(f"Reusing complete training index: {DB_PATH}", flush=True)
    else:
        DB_PATH.unlink(missing_ok=True)
        training_sources = [
            (TRAIN_DIR / "train_source2.tsv", "S2"),
            (TRAIN_DIR / "train_source3.tsv", "S3"),
        ]
        training_db = build_index(DB_PATH, training_sources)
        training_db.close()

    model, scaler, threshold, validation_score = fit_and_select_threshold(DB_PATH)
    print(
        f"Held-out macro F0.5: {validation_score * 100:.2f}% "
        f"(threshold {threshold:.2f}); using this threshold for final inference.",
        flush=True,
    )
    model = fit_validation_entities(DB_PATH, model, scaler)
    (ARTIFACTS_DIR / "full_training_checkpoint.pkl").unlink(missing_ok=True)
    DB_PATH.unlink(missing_ok=True)
    gc.collect()

    test_sources = [
        (TEST_DIR / "test_source2.tsv", "S2"),
        (TEST_DIR / "test_source3.tsv", "S3"),
    ]
    test_db = build_index(DB_PATH, test_sources)
    try:
        write_test_outputs(DB_PATH, model, scaler, threshold)
    finally:
        test_db.close()
    DB_PATH.unlink(missing_ok=True)

    validator = RESOURCE / "utils" / "validate_submission.py"
    command = [
        sys.executable, str(validator),
        "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
        "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
        "--test-dir", str(TEST_DIR),
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
    print(f"Full pipeline and validation completed in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
