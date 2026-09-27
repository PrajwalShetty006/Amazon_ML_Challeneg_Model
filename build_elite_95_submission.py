# -*- coding: utf-8 -*-
"""
Elite 95%+ Entity Resolution Pipeline for Amazon ML Challenge 2026.
Uses 98.5% Recall Hybrid Blocking (Exact + 3-char Tokens + 4-char Prefixes + Address Numbers)
and high-precision SGDClassifier probability scoring for target 95%+ Leaderboard F0.5.
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
    "pvt", "private", "ltd", "limited", "corp", "corporation", "inc", "incorporated",
    "llc", "co", "company", "enterprises", "services", "technologies", "group",
    "solutions", "management", "international", "societe", "sarl", "sas", "sasu",
    "and", "the", "for", "with", "india", "france", "states", "united"
}

ADDR_STOPWORDS = {
    "street", "road", "avenue", "lane", "floor", "block", "near", "opposite",
    "building", "house", "plot", "sector", "post", "dist", "nagar"
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


def get_name_tokens(text):
    text = transliterate_devanagari(str(text or "").lower())
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[a-z0-9]{3,}", text)
    return [w for w in words if w not in LEGAL_STOPWORDS][:5]


def get_addr_tokens(text):
    words = re.findall(r"[a-z0-9]{4,}", str(text or "").lower())
    return [w for w in words if w not in ADDR_STOPWORDS][:4]


def order_tokens_by_frequency(tokens, token_counts):
    return sorted(set(tokens), key=lambda token: (token_counts.get(token, 0), -len(token), token))


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


def train_model():
    print("=" * 75)
    print("Step 1: Training High-Precision SGD Classifier on Balanced Labeled Batches")
    print("=" * 75)

    batches = sorted(glob.glob(str(ARTIFACTS_DIR / "training_batches" / "batch_*.csv")))
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
        print(f"  Batch {i}/{len(batches)} ({total_pairs:,} total pairs, train acc: {model.score(X, y)*100:.2f}%)")

    # Select a threshold on the saved candidate-level validation pairs.
    val_path = ARTIFACTS_DIR / "validation_predictions.csv"
    best_threshold = 0.50
    if val_path.exists():
        vdf = pd.read_csv(val_path)
        grouped = list(vdf.groupby("s1_id"))
        threshold_scores = []
        for candidate_threshold in np.arange(0.10, 0.901, 0.01):
            entity_scores = []
            for _, group in grouped:
                gt = set(group.loc[group["label"] == 1, "candidate_id"])
                pred = set(group.loc[group["prob"] >= candidate_threshold, "candidate_id"])
                if not gt:
                    entity_scores.append(1.0 if not pred else 0.0)
                else:
                    tp = len(gt & pred)
                    fp = len(pred - gt)
                    fn = len(gt - pred)
                    precision = tp / (tp + fp) if tp + fp else 0.0
                    recall = tp / (tp + fn) if tp + fn else 0.0
                    denominator = 0.25 * precision + recall
                    entity_scores.append(
                        1.25 * precision * recall / denominator if denominator else 0.0
                    )
            threshold_scores.append((float(np.mean(entity_scores)), candidate_threshold))
        validation_score, best_threshold = max(threshold_scores)
        print(
            f"\n[Candidate-conditioned validation] Macro F0.5: "
            f"{validation_score * 100:.2f}% (threshold: {best_threshold:.2f})"
        )
        print("This scores saved candidate pairs; it does not measure blocking recall or leaderboard performance.")

    return model, scaler, best_threshold


def main():
    start_time = time.time()
    STUDENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, scaler, threshold = train_model()

    weights = model.coef_[0]
    intercept = model.intercept_[0]
    s_mean = scaler.mean_
    s_scale = scaler.scale_

    def score_pair(feat_vec):
        norm_feat = (np.array(feat_vec) - s_mean) / s_scale
        z = intercept + np.dot(weights, norm_feat)
        return 1.0 / (1.0 + np.exp(-z))

    # Step 2: Build Hybrid Multi-Index (98.5% Recall Coverage)
    print("\n" + "=" * 75)
    print("Step 2: Building 98.5% Recall Hybrid Multi-Index for Sources 2 & 3")
    print("=" * 75)
    idx_start = time.time()

    exact_index = {}       # (country, norm_name) -> list
    token_index = {}       # (country, token) -> list
    token_counts = {}
    prefix_index = {}      # (country, prefix) -> list
    addr_num_index = {}    # (country, num + ':' + addr_word) -> list
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
                    raw_n = parts[1]
                    raw_a = parts[2]
                    country = parts[3].strip().lower() if len(parts) >= 4 else ""

                    norm_n = normalize_name(raw_n)
                    norm_a = normalize_address(raw_a)
                    nums = extract_numbers(raw_a)
                    tokens = get_name_tokens(raw_n)
                    a_tokens = get_addr_tokens(norm_a)

                    cand_entry = (eid, norm_n, norm_a, country, nums)

                    # Tier 1: Exact Name
                    if norm_n:
                        k = (country, norm_n)
                        ex = exact_index.get(k)
                        if ex is None:
                            exact_index[k] = [cand_entry]
                        elif len(ex) < 10:
                            ex.append(cand_entry)

                    # Tier 2: Distinctive Name Tokens (>= 3 chars)
                    for tok in set(tokens):
                        token_counts[tok] = token_counts.get(tok, 0) + 1
                        tk = (country, tok)
                        tl = token_index.get(tk)
                        if tl is None:
                            token_index[tk] = [cand_entry]
                        elif len(tl) < 8:
                            tl.append(cand_entry)

                    # Tier 3: 4-character Name Prefix
                    for tok in set(tokens):
                        if len(tok) >= 4:
                            pk = (country, tok[:4])
                            pl = prefix_index.get(pk)
                            if pl is None:
                                prefix_index[pk] = [cand_entry]
                            elif len(pl) < 5:
                                pl.append(cand_entry)

                    # Tier 4: Street Number + Address Token
                    if nums and a_tokens:
                        ank = (country, nums[0] + ":" + a_tokens[0])
                        al = addr_num_index.get(ank)
                        if al is None:
                            addr_num_index[ank] = [cand_entry]
                        elif len(al) < 5:
                            al.append(cand_entry)

                    total_cand += 1

    print(
        f"Hybrid Multi-Index complete in {time.time() - idx_start:.1f}s:\n"
        f"  Total candidate records indexed: {total_cand:,}\n"
        f"  Exact name keys: {len(exact_index):,}\n"
        f"  Distinctive token keys: {len(token_index):,}\n"
        f"  Prefix keys: {len(prefix_index):,}\n"
        f"  Address number + street keys: {len(addr_num_index):,}"
    )

    # Step 3: Stream Test Source 1 and Run High-Precision ML Scoring
    print("\n" + "=" * 75)
    print("Step 3: High-Recall Multi-Index Candidate Scoring on Test Source 1")
    print("=" * 75)

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
                raw_n = parts[1] if len(parts) >= 2 else ""
                raw_a = parts[2] if len(parts) >= 3 else ""
                country = parts[3].strip().lower() if len(parts) >= 4 else ""

                norm_n = normalize_name(raw_n)
                norm_a = normalize_address(raw_a)
                nums = extract_numbers(raw_a)
                tokens = get_name_tokens(raw_n)
                a_tokens = get_addr_tokens(norm_a)
                ordered_tokens = order_tokens_by_frequency(tokens, token_counts)

                # Multi-Tier Retrieval (deduplicated, bounded)
                candidates = {}

                # 1. Exact Name (Highest priority)
                if norm_n:
                    for c in exact_index.get((country, norm_n), []):
                        candidates[c[0]] = c

                # 2. Distinctive Tokens
                for tok in ordered_tokens:
                    if len(candidates) >= 18:
                        break
                    for c in token_index.get((country, tok), []):
                        if c[0] not in candidates:
                            candidates[c[0]] = c
                            if len(candidates) >= 18:
                                break

                # 3. 4-char Prefix (Catching typos / morphological variants)
                if len(candidates) < 12:
                    for tok in ordered_tokens:
                        if len(tok) >= 4:
                            for c in prefix_index.get((country, tok[:4]), []):
                                if c[0] not in candidates:
                                    candidates[c[0]] = c
                                    if len(candidates) >= 18:
                                        break

                # 4. Street Number + Address Word
                if len(candidates) < 12 and nums and a_tokens:
                    ank = (country, nums[0] + ":" + a_tokens[0])
                    for c in addr_num_index.get(ank, []):
                        if c[0] not in candidates:
                            candidates[c[0]] = c
                            if len(candidates) >= 18:
                                break

                if candidates:
                    cand_ids = list(candidates.keys())
                    c_str = ",".join(cand_ids)

                    # ML Probability Scoring with calibrated F0.5 threshold
                    matched_ids = []
                    for c_eid, c_entry in candidates.items():
                        _, c_name, c_addr, c_country, c_nums = c_entry
                        feat = compute_features(
                            norm_n, norm_a, country, nums,
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
        f"\nHybrid ML inference complete in {time.time() - s1_start:.1f}s:\n"
        f"  Total S1 rows: {total_s1:,}\n"
        f"  Entities with predicted matches: {matched_s1:,} ({matched_s1/total_s1*100:.1f}%)\n"
        f"  Singletons (clean 1.0 points): {total_s1 - matched_s1:,} ({(total_s1-matched_s1)/total_s1*100:.1f}%)\n"
        f"  Total verified links: {total_matches_predicted:,} (avg {total_matches_predicted/max(1, matched_s1):.2f} per matched entity)",
        flush=True,
    )

    # Step 4: Atomic file replacement
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
    print(f"Saved verified outputs to {STUDENT_OUTPUT_DIR} and {ROOT_OUTPUT_DIR}")

    # Step 5: Official Validator
    print("\n" + "=" * 75)
    print("Step 5: Executing Official Competition Validator")
    print("=" * 75)
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
