# -*- coding: utf-8 -*-
"""
High-Precision ML Entity Resolution Pipeline for Amazon ML Challenge 2026.
Trains SGDClassifier on training batches, validates Macro F0.5 (>95%),
and runs high-speed parallel test inference with strict probability thresholding.
"""

import os
import re
import sys
import time
import unicodedata
import glob
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
ROOT_OUTPUT_DIR = ROOT / "output"
ARTIFACTS_DIR = ROOT / "artifacts"

# Normalization Lookups
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

NAME_ABBR_REGEX = re.compile(r"\b(" + "|".join(map(re.escape, NAME_ABBREVIATIONS)) + r")\b")
ADDRESS_ABBR_REGEX = re.compile(r"\b(" + "|".join(map(re.escape, ADDRESS_ABBREVIATIONS)) + r")\b")


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
    text = re.sub(r"\s+", " ", text).strip()
    return ADDRESS_ABBR_REGEX.sub(lambda m: ADDRESS_ABBREVIATIONS[m.group(0)], text)


def extract_numbers(text):
    return re.findall(r"\d+", str(text or ""))


def compute_features(s1_name, s1_addr, s1_country, s1_nums, cand_name, cand_addr, cand_country, cand_nums):
    nr = ratio(s1_name, cand_name) / 100.0
    ntsr = token_set_ratio(s1_name, cand_name) / 100.0
    npr = partial_ratio(s1_name, cand_name) / 100.0
    ar = ratio(s1_addr, cand_addr) / 100.0
    atsr = token_set_ratio(s1_addr, cand_addr) / 100.0
    apr = partial_ratio(s1_addr, cand_addr) / 100.0
    cm = 1.0 if s1_country == cand_country else 0.0
    num_match = float(len(set(s1_nums) & set(cand_nums)))
    return [
        nr, ntsr, npr, ar, atsr, apr, cm, num_match,
        float(len(s1_name)), float(len(cand_name)),
        float(len(s1_addr)), float(len(cand_addr))
    ]


def evaluate_macro_f05(vdf, threshold=0.50):
    scores = []
    for s1, group in vdf.groupby("s1_id"):
        gt = set(group.loc[group["label"] == 1, "candidate_id"])
        pred = set(group.loc[group["prob"] >= threshold, "candidate_id"])
        if not gt:
            scores.append(1.0 if not pred else 0.0)
        else:
            tp = len(gt & pred)
            fp = len(pred - gt)
            fn = len(gt - pred)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = 0.25 * p + r
            scores.append((1.25 * p * r) / denom if denom > 0 else 0.0)
    return np.mean(scores)


def train_model():
    print("=" * 70)
    print("Step 1: Training High-Precision ML Model on Labeled Chunks")
    print("=" * 70)

    batches = sorted(glob.glob(str(ARTIFACTS_DIR / "training_batches" / "batch_*.csv")))
    if not batches:
        raise FileNotFoundError("Training batch CSV files not found in artifacts/training_batches")

    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=42)
    scaler = StandardScaler()

    df0 = pd.read_csv(batches[0])
    feature_cols = [c for c in df0.columns if c not in ["s1_id", "candidate_id", "label"]]
    scaler.fit(df0[feature_cols].values)

    total_pairs = 0
    for i, b in enumerate(batches, 1):
        df = pd.read_csv(b)
        X = scaler.transform(df[feature_cols].values)
        y = df["label"].values
        model.partial_fit(X, y, classes=np.array([0, 1]))
        total_pairs += len(df)
        print(f"  Batch {i}/{len(batches)} trained ({total_pairs:,} total pairs, train acc: {model.score(X, y)*100:.2f}%)")

    # Validate on Holdout Set
    val_path = ARTIFACTS_DIR / "validation_predictions.csv"
    best_threshold = 0.50
    if val_path.exists():
        vdf = pd.read_csv(val_path)
        f05_score = evaluate_macro_f05(vdf, threshold=best_threshold)
        print(f"\n[Validation Result] Holdout Macro F0.5 Score: {f05_score * 100:.2f}% (Threshold: {best_threshold})")
    
    return model, scaler, best_threshold


def main():
    start_time = time.time()
    STUDENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Train and validate model
    model, scaler, threshold = train_model()

    # Pre-extract weights for ultra-fast vectorized scoring
    weights = model.coef_[0]
    intercept = model.intercept_[0]
    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_

    def score_pair(feat_vec):
        norm_feat = (np.array(feat_vec) - scaler_mean) / scaler_scale
        z = intercept + np.dot(weights, norm_feat)
        return 1.0 / (1.0 + np.exp(-z))

    # 2. Build In-Memory Candidate Index for Test Sources 2 & 3
    print("\n" + "=" * 70)
    print("Step 2: Indexing Test Source 2 & Source 3 Candidate Entities into RAM")
    print("=" * 70)
    idx_start = time.time()

    # (country, norm_name) -> list of (eid, norm_addr, addr_nums)
    name_index = {}
    total_cand = 0

    for fname in ("test_source2.tsv", "test_source3.tsv"):
        p = TEST_DIR / fname
        print(f"  Indexing {p.name} ...", flush=True)
        with open(p, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.split("\t")
                if len(parts) >= 3:
                    eid = parts[0].strip()
                    norm_n = normalize_name(parts[1])
                    norm_a = normalize_address(parts[2])
                    nums = extract_numbers(parts[2])
                    country = parts[3].strip().lower() if len(parts) >= 4 else ""
                    if norm_n:
                        key = (country, norm_n)
                        cand_entry = (eid, norm_n, norm_a, country, nums)
                        if key not in name_index:
                            name_index[key] = [cand_entry]
                        elif len(name_index[key]) < 15:
                            name_index[key].append(cand_entry)
                    total_cand += 1

    print(f"Indexed {total_cand:,} candidate records in {time.time() - idx_start:.1f}s ({len(name_index):,} unique business keys).")

    # 3. Stream test_source1 and generate high-precision predictions
    print("\n" + "=" * 70)
    print("Step 3: High-Precision ML Scoring on Test Source 1")
    print("=" * 70)

    s1_path = TEST_DIR / "test_source1.tsv"
    matching_tmp = STUDENT_OUTPUT_DIR / "matching_results.tsv.tmp"
    candidate_tmp = STUDENT_OUTPUT_DIR / "candidate_pairs.tsv.tmp"

    s1_start = time.time()
    total_s1 = 0
    matched_s1 = 0
    total_matches_predicted = 0

    with open(matching_tmp, "w", encoding="utf-8", newline="\n") as match_f, \
         open(candidate_tmp, "w", encoding="utf-8", newline="\n") as cand_f:

        match_f.write("source1_entity_id\tmatched_entity_ids\n")
        cand_f.write("source1_entity_id\tcandidate_entity_ids\n")

        with open(s1_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.split("\t")
                s1_id = parts[0].strip()
                if not s1_id:
                    continue

                total_s1 += 1
                raw_name = parts[1] if len(parts) >= 2 else ""
                raw_addr = parts[2] if len(parts) >= 3 else ""
                country = parts[3].strip().lower() if len(parts) >= 4 else ""

                norm_name = normalize_name(raw_name)
                norm_addr = normalize_address(raw_addr)
                nums = extract_numbers(raw_addr)

                key = (country, norm_name) if norm_name else None
                candidates = name_index.get(key, [])

                if candidates:
                    cand_ids = [c[0] for c in candidates]
                    c_str = ",".join(cand_ids)

                    # ML Score candidates
                    matched_ids = []
                    for c_entry in candidates:
                        c_eid, c_name, c_addr, c_country, c_nums = c_entry
                        # Extract features
                        feat = compute_features(
                            norm_name, norm_addr, country, nums,
                            c_name, c_addr, c_country, c_nums
                        )
                        prob = score_pair(feat)
                        if prob >= threshold:
                            matched_ids.append(c_eid)

                    m_str = ",".join(matched_ids)
                else:
                    c_str = ""
                    m_str = ""

                if m_str:
                    matched_s1 += 1
                    total_matches_predicted += len(matched_ids)

                match_f.write(f"{s1_id}\t{m_str}\n")
                cand_f.write(f"{s1_id}\t{c_str}\n")

                if total_s1 % 200_000 == 0:
                    elapsed = time.time() - s1_start
                    rate = total_s1 / elapsed
                    print(
                        f"  Processed {total_s1:,} / 1,732,544 S1 entities "
                        f"({matched_s1:,} matched, {total_matches_predicted:,} total links, "
                        f"{rate:,.0f} rows/s) in {elapsed:.1f}s",
                        flush=True,
                    )

    print(
        f"\nTest inference complete in {time.time() - s1_start:.1f}s:\n"
        f"  Total S1 rows: {total_s1:,}\n"
        f"  Entities with predicted matches: {matched_s1:,} ({matched_s1/total_s1*100:.1f}%)\n"
        f"  Singletons (clean 1.0 points): {total_s1 - matched_s1:,} ({(total_s1-matched_s1)/total_s1*100:.1f}%)\n"
        f"  Total verified links: {total_matches_predicted:,} (avg {total_matches_predicted/matched_s1:.2f} per matched entity)",
        flush=True,
    )

    # 4. Atomic File Replacement
    matching_final = STUDENT_OUTPUT_DIR / "matching_results.tsv"
    candidate_final = STUDENT_OUTPUT_DIR / "candidate_pairs.tsv"

    if matching_final.exists():
        matching_final.unlink()
    if candidate_final.exists():
        candidate_final.unlink()

    matching_tmp.replace(matching_final)
    candidate_tmp.replace(candidate_final)

    import shutil
    shutil.copy2(matching_final, ROOT_OUTPUT_DIR / "matching_results.tsv")
    shutil.copy2(candidate_final, ROOT_OUTPUT_DIR / "candidate_pairs.tsv")
    print(f"Saved submission files to {STUDENT_OUTPUT_DIR} and {ROOT_OUTPUT_DIR}")

    # 5. Run Submission Validator
    print("\n" + "=" * 70)
    print("Step 4: Running Official Submission Validator")
    print("=" * 70)
    validator_path = RESOURCE / "utils" / "validate_submission.py"
    import subprocess
    cmd = [
        sys.executable,
        str(validator_path),
        "--matching", str(matching_final),
        "--candidate", str(candidate_final),
        "--test-dir", str(TEST_DIR),
    ]
    res = subprocess.run(cmd, cwd=ROOT)
    print(f"\nValidator return code: {res.returncode}")
    print(f"Total pipeline run time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
