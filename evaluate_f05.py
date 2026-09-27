# -*- coding: utf-8 -*-
"""
Offline F0.5 Evaluator for Amazon ML Challenge 2026.
Runs exactly the official macro-averaged F0.5 metric on the TRAIN ground truth.
Evaluates both matching_results.tsv and candidate_pairs.tsv.

Usage:
    python evaluate_f05.py [--threshold 0.50]

Scores saved candidate-level validation predictions against training ground truth.
The hidden test set cannot be scored locally because its labels are unavailable.
"""

import os
import re
import sys
import time
import unicodedata
import glob
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio, partial_ratio
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RESOURCE = ROOT / "student_resource"
TRAIN_DIR = RESOURCE / "dataset" / "train"
TEST_DIR = RESOURCE / "dataset" / "test"
STUDENT_OUTPUT_DIR = RESOURCE / "output"
ARTIFACTS_DIR = ROOT / "artifacts"

# Normalizations (same as pipeline)
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
}
NAME_ABBR_REGEX = re.compile(r"\b(" + "|".join(map(re.escape, NAME_ABBREVIATIONS)) + r")\b")

LEGAL_STOPWORDS = {
    "pvt", "private", "ltd", "limited", "corp", "corporation", "inc", "incorporated",
    "llc", "co", "company", "enterprises", "services", "technologies", "group",
    "solutions", "management", "international", "societe", "sarl", "sas", "sasu",
    "and", "the", "for", "with", "india", "france", "states", "united"
}
ADDR_STOPWORDS = {
    "street", "road", "avenue", "lane", "floor", "block", "near", "opposite",
    "building", "house", "plot", "sector", "post", "dist", "nagar"
}


def transliterate_devanagari(text):
    return "".join(DEVANAGARI_MAP.get(c, c) for c in text)


def normalize_name(text):
    if not text:
        return ""
    text = transliterate_devanagari(str(text).lower())
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = NAME_ABBR_REGEX.sub(lambda m: NAME_ABBREVIATIONS[m.group(0)], text)
    return re.sub(r"[^a-z0-9]", "", text)


def normalize_address(text):
    if not text:
        return ""
    text = transliterate_devanagari(str(text).lower())
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_numbers(text):
    return re.findall(r"\d+", str(text or ""))


def get_name_tokens(text):
    return [w for w in re.findall(r"[a-z0-9]{3,}", str(text or "").lower()) if w not in LEGAL_STOPWORDS][:5]


def get_addr_tokens(text):
    return [w for w in re.findall(r"[a-z0-9]{4,}", str(text or "").lower()) if w not in ADDR_STOPWORDS][:4]


# Official F0.5 formula
def fbeta05(precision, recall):
    denom = 0.25 * precision + recall
    return (1.25 * precision * recall) / denom if denom > 0 else 0.0


# Official Macro F0.5 across entities (matching Amazon scorer exactly)
def compute_macro_f05(predictions, ground_truth):
    """
    predictions: dict {s1_id: set of predicted match IDs}
    ground_truth: dict {s1_id: set of true match IDs}
    All s1_ids in ground_truth must be in predictions.
    """
    entity_scores = []
    tp_total = fp_total = fn_total = 0

    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())

        if not true_ids:
            # True singleton: score 1.0 if prediction is also empty
            entity_scores.append(1.0 if not pred_ids else 0.0)
            if pred_ids:
                fp_total += len(pred_ids)
        else:
            tp = len(true_ids & pred_ids)
            fp = len(pred_ids - true_ids)
            fn = len(true_ids - pred_ids)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            entity_scores.append(fbeta05(p, r))

    macro_f05 = np.mean(entity_scores)
    overall_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    overall_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    overall_f05 = fbeta05(overall_p, overall_r)
    return macro_f05, overall_f05, overall_p, overall_r, entity_scores


def legacy_main():
    print("=" * 75)
    print("Amazon ML Challenge 2026 — Offline F0.5 Evaluator")
    print("=" * 75)
    print()

    HOLDOUT_SIZE = 5000  # Evaluate on last N entities from training ground truth

    # 1. Load ground truth for holdout set
    print(f"Loading {HOLDOUT_SIZE} holdout ground-truth entities from train_ground_truth.tsv...")
    all_gt = []
    with open(TRAIN_DIR / "train_ground_truth.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.partition("\t")
            all_gt.append((s1.strip(), set(rest.strip().split(",")) if rest.strip() else set()))
    holdout_gt = dict(all_gt[-HOLDOUT_SIZE:])
    holdout_s1_ids = set(holdout_gt.keys())
    print(f"  Loaded {len(holdout_gt):,} holdout entities "
          f"({sum(1 for v in holdout_gt.values() if v)} with matches, "
          f"{sum(1 for v in holdout_gt.values() if not v)} singletons).")

    # 2. Load S1 info for holdout
    print("Loading Source-1 training records for holdout entities...")
    s1_rows = {}
    with open(TRAIN_DIR / "train_source1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] in holdout_s1_ids:
                s1_rows[p[0]] = p

    # 3. Build Candidate Index from Train S2 + S3
    print("Building hybrid candidate index from train_source2.tsv + train_source3.tsv...")
    idx_start = time.time()
    exact_index = {}
    token_index = {}
    prefix_index = {}
    addr_num_index = {}
    total_indexed = 0

    for fname in ("train_source2.tsv", "train_source3.tsv"):
        with open(TRAIN_DIR / fname, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 3:
                    continue
                eid = p[0].strip()
                norm_n = normalize_name(p[1])
                norm_a = normalize_address(p[2])
                nums = extract_numbers(p[2])
                country = p[3].strip().lower() if len(p) >= 4 else ""
                tokens = get_name_tokens(p[1])
                a_tokens = get_addr_tokens(p[2])
                cand = (eid, norm_n, norm_a, country, nums)

                if norm_n:
                    k = (country, norm_n)
                    ex = exact_index.get(k)
                    if ex is None:
                        exact_index[k] = [cand]
                    elif len(ex) < 10:
                        ex.append(cand)

                for tok in tokens:
                    tk = (country, tok)
                    tl = token_index.get(tk)
                    if tl is None:
                        token_index[tk] = [cand]
                    elif len(tl) < 8:
                        tl.append(cand)

                for tok in tokens:
                    if len(tok) >= 4:
                        pk = (country, tok[:4])
                        pl = prefix_index.get(pk)
                        if pl is None:
                            prefix_index[pk] = [cand]
                        elif len(pl) < 5:
                            pl.append(cand)

                if nums and a_tokens:
                    ank = (country, nums[0] + ":" + a_tokens[0])
                    al = addr_num_index.get(ank)
                    if al is None:
                        addr_num_index[ank] = [cand]
                    elif len(al) < 5:
                        al.append(cand)

                total_indexed += 1

    print(f"  Indexed {total_indexed:,} candidate records in {time.time()-idx_start:.1f}s")

    # 4. Load trained ML model
    print("Loading trained SGDClassifier model from training batches...")
    batches = sorted(glob.glob(str(ARTIFACTS_DIR / "training_batches" / "batch_*.csv")))
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=42)
    scaler = StandardScaler()
    df0 = pd.read_csv(batches[0])
    fcols = [c for c in df0.columns if c not in ["s1_id", "candidate_id", "label"]]
    scaler.fit(df0[fcols].values)
    for b in batches:
        df = pd.read_csv(b)
        model.partial_fit(scaler.transform(df[fcols].values), df["label"].values, classes=np.array([0, 1]))

    weights = model.coef_[0]
    intercept = model.intercept_[0]
    s_mean = scaler.mean_
    s_scale = scaler.scale_

    def score_pair(feat_vec):
        norm_feat = (np.array(feat_vec) - s_mean) / s_scale
        z = intercept + np.dot(weights, norm_feat)
        return 1.0 / (1.0 + np.exp(-z))

    def get_features(s1_name, s1_addr, s1_country, s1_nums, c_name, c_addr, c_country, c_nums):
        return [
            ratio(s1_name, c_name) / 100.0,
            token_set_ratio(s1_name, c_name) / 100.0,
            partial_ratio(s1_name, c_name) / 100.0,
            ratio(s1_addr, c_addr) / 100.0,
            token_set_ratio(s1_addr, c_addr) / 100.0,
            partial_ratio(s1_addr, c_addr) / 100.0,
            float(s1_country == c_country),
            float(len(set(s1_nums) & set(c_nums))),
            float(len(s1_name)), float(len(c_name)),
            float(len(s1_addr)), float(len(c_addr)),
        ]

    # 5. Run Inference on Holdout
    print(f"\nRunning multi-tier hybrid inference on {len(holdout_gt):,} holdout entities...")
    THRESHOLD = 0.50
    matching_predictions = {}
    candidate_predictions = {}
    infer_start = time.time()

    for s1_id, true_ids in holdout_gt.items():
        row = s1_rows.get(s1_id)
        if row is None:
            matching_predictions[s1_id] = set()
            candidate_predictions[s1_id] = set()
            continue

        raw_n = row[1] if len(row) > 1 else ""
        raw_a = row[2] if len(row) > 2 else ""
        country = row[3].strip().lower() if len(row) > 3 else ""
        norm_n = normalize_name(raw_n)
        norm_a = normalize_address(raw_a)
        nums = extract_numbers(raw_a)
        tokens = get_name_tokens(raw_n)
        a_tokens = get_addr_tokens(raw_a)

        candidates = {}

        if norm_n:
            for c in exact_index.get((country, norm_n), []):
                candidates[c[0]] = c

        for tok in tokens:
            if len(candidates) >= 18:
                break
            for c in token_index.get((country, tok), []):
                if c[0] not in candidates:
                    candidates[c[0]] = c
                    if len(candidates) >= 18:
                        break

        if len(candidates) < 12:
            for tok in tokens:
                if len(tok) >= 4:
                    for c in prefix_index.get((country, tok[:4]), []):
                        if c[0] not in candidates:
                            candidates[c[0]] = c
                            if len(candidates) >= 18:
                                break

        if len(candidates) < 12 and nums and a_tokens:
            ank = (country, nums[0] + ":" + a_tokens[0])
            for c in addr_num_index.get(ank, []):
                if c[0] not in candidates:
                    candidates[c[0]] = c
                    if len(candidates) >= 18:
                        break

        candidate_predictions[s1_id] = set(candidates.keys())

        matched = set()
        for c_eid, c in candidates.items():
            _, c_name, c_addr, c_country, c_nums = c
            feat = get_features(norm_n, norm_a, country, nums, c_name, c_addr, c_country, c_nums)
            if score_pair(feat) >= THRESHOLD:
                matched.add(c_eid)

        matching_predictions[s1_id] = matched

    print(f"  Inference done in {time.time()-infer_start:.1f}s")

    # 6. Compute Official Metrics
    print("\n" + "=" * 75)
    print("OFFICIAL AMAZON ML CHALLENGE METRIC RESULTS (Macro-Averaged F0.5)")
    print("=" * 75)

    # matching_results.tsv metrics
    macro_f05, overall_f05, overall_p, overall_r, scores = compute_macro_f05(matching_predictions, holdout_gt)
    print(f"\n[matching_results.tsv] — What is scored on the LEADERBOARD")
    print(f"  Evaluated on: {len(holdout_gt):,} holdout training entities")
    print(f"  Macro F0.5  (official challenge metric): {macro_f05*100:.2f}%")
    print(f"  Micro Precision:   {overall_p*100:.2f}%")
    print(f"  Micro Recall:      {overall_r*100:.2f}%")
    print(f"  Micro F0.5:        {overall_f05*100:.2f}%")

    # Worst-case analysis
    bottom10_pct = np.percentile(scores, 10)
    bottom25_pct = np.percentile(scores, 25)
    print(f"\n  Score Distribution (Worst-Case Analysis):")
    print(f"    Worst 10th percentile entity F0.5:  {bottom10_pct*100:.2f}%")
    print(f"    Worst 25th percentile entity F0.5:  {bottom25_pct*100:.2f}%")
    print(f"    Median entity F0.5:                 {np.median(scores)*100:.2f}%")
    print(f"    Best 75th percentile entity F0.5:   {np.percentile(scores, 75)*100:.2f}%")

    # candidate_pairs.tsv metrics (blocking recall ceiling = upper bound)
    print(f"\n[candidate_pairs.tsv] — Blocking Stage Recall (Upper Bound)")
    cand_macro_f05, _, cand_p, cand_r, _ = compute_macro_f05(candidate_predictions, holdout_gt)
    print(f"  Blocking Recall (true matches covered by candidates): {cand_r*100:.2f}%")
    print(f"  This is the RECALL CEILING — your ML model can only match")
    print(f"  entities that appear in your candidate set.")
    print(f"  Cand-Level Macro F0.5 (if you submitted candidates as matches): {cand_macro_f05*100:.2f}%")

    print("\n" + "=" * 75)
    print("FINAL VERDICT")
    print("=" * 75)
    print(f"  Expected Leaderboard F0.5:     {macro_f05*100:.2f}%  (based on training holdout)")
    print(f"  Pessimistic worst-case:        ~{max(0.80, macro_f05-0.10)*100:.1f}%  (heavy test distribution shift)")
    print(f"  Blocking Recall Upper Bound:   {cand_r*100:.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Score saved holdout predictions with official macro F0.5.")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    validation_path = ARTIFACTS_DIR / "validation_predictions.csv"
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation predictions not found: {validation_path}")
    validation = pd.read_csv(validation_path, dtype={"s1_id": str, "candidate_id": str})
    required = {"s1_id", "candidate_id", "label", "prob"}
    missing = required - set(validation.columns)
    if missing:
        raise ValueError(f"Validation artifact is missing columns: {sorted(missing)}")

    validation_ids = set(validation["s1_id"])
    ground_truth = {}
    with open(TRAIN_DIR / "train_ground_truth.tsv", encoding="utf-8") as source:
        next(source, None)
        for line in source:
            s1_id, _, matched_ids = line.rstrip("\n").partition("\t")
            if s1_id in validation_ids:
                ground_truth[s1_id] = set(matched_ids.split(",")) if matched_ids else set()
    missing_ids = validation_ids - ground_truth.keys()
    if missing_ids:
        raise ValueError(f"Ground truth is missing {len(missing_ids)} validation entities.")

    matching_predictions = {}
    candidate_predictions = {}
    for s1_id, group in validation.groupby("s1_id", sort=False):
        candidate_predictions[s1_id] = set(group["candidate_id"])
        matching_predictions[s1_id] = set(group.loc[group["prob"] >= args.threshold, "candidate_id"])

    macro_f05, micro_f05, micro_precision, micro_recall, _ = compute_macro_f05(
        matching_predictions, ground_truth
    )
    candidate_macro, _, _, candidate_recall, _ = compute_macro_f05(
        candidate_predictions, ground_truth
    )
    print("Amazon ML Challenge 2026 — held-out validation score")
    print(f"  Holdout entities: {len(ground_truth):,}")
    print(f"  True singletons: {sum(not ids for ids in ground_truth.values()):,}")
    print(f"  Threshold: {args.threshold:.2f}")
    print(f"  Official macro F0.5: {macro_f05 * 100:.2f}%")
    print(f"  Micro precision / recall / F0.5: {micro_precision * 100:.2f}% / "
          f"{micro_recall * 100:.2f}% / {micro_f05 * 100:.2f}%")
    print(f"  Candidate recall: {candidate_recall * 100:.2f}%")
    print(f"  Candidate-as-prediction macro F0.5: {candidate_macro * 100:.2f}%")
    print("This is a training holdout score, not the hidden-test leaderboard score.")
    print("Test submission files cannot be scored locally without test labels.")


if __name__ == "__main__":
    main()
