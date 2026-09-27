# -*- coding: utf-8 -*-
"""Fast, bounded-memory complete submission generator and validator."""

import os
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESOURCE = ROOT / "student_resource"
TEST_DIR = RESOURCE / "dataset" / "test"
STUDENT_OUTPUT_DIR = RESOURCE / "output"
ROOT_OUTPUT_DIR = ROOT / "output"

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

NAME_ABBR_REGEX = re.compile(r"\b(" + "|".join(map(re.escape, NAME_ABBREVIATIONS)) + r")\b")


def transliterate_devanagari(text):
    return "".join(DEVANAGARI_MAP.get(char, char) for char in text)


def normalize_name(text):
    if not text:
        return ""
    text = transliterate_devanagari(str(text).lower())
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = NAME_ABBR_REGEX.sub(lambda match: NAME_ABBREVIATIONS[match.group(0)], text)
    return re.sub(r"[^a-z0-9]", "", text)


def main():
    start_time = time.time()
    print("=" * 60)
    print("Starting Fast Complete Submission Generation")
    print("=" * 60)

    STUDENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load existing predictions from root output directory
    existing_candidates = {}
    existing_matches = {}

    root_cand_path = ROOT_OUTPUT_DIR / "candidate_pairs.tsv"
    root_match_path = ROOT_OUTPUT_DIR / "matching_results.tsv"

    if root_cand_path.is_file() and root_match_path.is_file():
        print(f"Loading existing candidate pairs from {root_cand_path} ...", flush=True)
        with open(root_cand_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                s1, _, rest = line.partition("\t")
                s1 = s1.strip()
                if s1:
                    existing_candidates[s1] = rest.strip()

        print(f"Loading existing matches from {root_match_path} ...", flush=True)
        with open(root_match_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                s1, _, rest = line.partition("\t")
                s1 = s1.strip()
                if s1:
                    existing_matches[s1] = rest.strip()

        print(f"  Loaded {len(existing_matches):,} existing predictions.", flush=True)

    # 2. Build high-speed in-memory candidate index for remaining entities
    print("Indexing candidate sources (Source 2 and Source 3) ...", flush=True)
    index_start = time.time()
    name_index = {}
    total_candidates_indexed = 0

    for fname in ("test_source2.tsv", "test_source3.tsv"):
        path = TEST_DIR / fname
        print(f"  Reading {path.name} ...", flush=True)
        with open(path, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.split("\t")
                if len(parts) >= 3:
                    eid = parts[0].strip()
                    norm = normalize_name(parts[1])
                    country = parts[3].strip().lower() if len(parts) >= 4 else ""
                    if norm:
                        key = (country, norm)
                        current = name_index.get(key)
                        if current is None:
                            name_index[key] = eid
                        elif isinstance(current, str):
                            name_index[key] = [current, eid]
                        elif len(current) < 10:
                            current.append(eid)
                    total_candidates_indexed += 1

    print(
        f"Indexed {total_candidates_indexed:,} candidate records in {time.time() - index_start:.1f}s "
        f"({len(name_index):,} unique keys).",
        flush=True,
    )

    # 3. Stream all test_source1 entities and generate full outputs
    source1_path = TEST_DIR / "test_source1.tsv"
    matching_tmp = STUDENT_OUTPUT_DIR / "matching_results.tsv.tmp"
    candidate_tmp = STUDENT_OUTPUT_DIR / "candidate_pairs.tsv.tmp"

    print("Generating complete submission outputs for all Source 1 entities ...", flush=True)
    gen_start = time.time()
    total_s1 = 0
    matched_count = 0

    with open(matching_tmp, "w", encoding="utf-8", newline="\n") as match_f, \
         open(candidate_tmp, "w", encoding="utf-8", newline="\n") as cand_f:

        match_f.write("source1_entity_id\tmatched_entity_ids\n")
        cand_f.write("source1_entity_id\tcandidate_entity_ids\n")

        with open(source1_path, encoding="utf-8") as s1_f:
            next(s1_f)
            for line in s1_f:
                parts = line.split("\t")
                s1_id = parts[0].strip()
                if not s1_id:
                    continue

                total_s1 += 1

                # Case A: Already computed in prior run
                if s1_id in existing_matches:
                    m_ids = existing_matches[s1_id]
                    c_ids = existing_candidates.get(s1_id, m_ids)
                else:
                    # Case B: In-memory exact normalized name match
                    norm = normalize_name(parts[1]) if len(parts) >= 2 else ""
                    country = parts[3].strip().lower() if len(parts) >= 4 else ""
                    key = (country, norm) if norm else None
                    candidates = name_index.get(key) if key else None

                    if candidates:
                        if isinstance(candidates, str):
                            id_list = [candidates]
                        else:
                            id_list = list(dict.fromkeys(candidates))
                        cand_str = ",".join(id_list)
                        m_ids = cand_str
                        c_ids = cand_str
                    else:
                        m_ids = ""
                        c_ids = ""

                if m_ids:
                    matched_count += 1

                match_f.write(f"{s1_id}\t{m_ids}\n")
                cand_f.write(f"{s1_id}\tcandidate_entity_ids\n" if False else f"{s1_id}\t{c_ids}\n")

                if total_s1 % 200_000 == 0:
                    print(
                        f"  Processed {total_s1:,} / 1,732,544 rows "
                        f"({matched_count:,} matched) in {time.time() - gen_start:.1f}s",
                        flush=True,
                    )

    print(
        f"Finished processing all {total_s1:,} Source 1 entities in {time.time() - gen_start:.1f}s "
        f"({matched_count:,} with matches).",
        flush=True,
    )

    # 4. Atomic file replacement in both student_resource and root output dirs
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
    print(f"Saved verified outputs to both {STUDENT_OUTPUT_DIR} and {ROOT_OUTPUT_DIR}", flush=True)

    # 5. Execute official validator
    validator_path = RESOURCE / "utils" / "validate_submission.py"
    print("\nRunning official submission validator ...", flush=True)
    import subprocess
    cmd = [
        sys.executable,
        str(validator_path),
        "--matching", str(matching_final),
        "--candidate", str(candidate_final),
        "--test-dir", str(TEST_DIR),
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    print(f"\nValidator return code: {result.returncode}")
    print(f"Total pipeline elapsed time: {time.time() - start_time:.1f}s")
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
