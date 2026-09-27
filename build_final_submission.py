# -*- coding: utf-8 -*-
"""
Final Submission Builder for Amazon ML Challenge 2026.
Four blocking tiers designed around actual data failure modes:
  Tier 1: Normalized name token inverted index (uncapped, country-agnostic)
  Tier 2: Domain/website normalization -> business name extraction
  Tier 3: TF-IDF character 3-gram cosine similarity (catches multilingual, aliases)
  Tier 4: Address number + first 4-char addr token

Targets: Blocking Recall >= 90%, Final F0.5 >= 0.88 on leaderboard.
"""

import os, re, sys, time, glob, unicodedata, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio, partial_ratio, WRatio
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
RESOURCE = ROOT / "student_resource"
TRAIN_DIR = RESOURCE / "dataset" / "train"
TEST_DIR = RESOURCE / "dataset" / "test"
OUT_DIR = RESOURCE / "output"
ARTIFACTS_DIR = ROOT / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TFIDF_TOP_K = 15
TOKEN_TOP_K  = 25
MATCH_THRESHOLD = 0.45

# =================== Normalization ===================

DEVANAGARI_MAP = {
    "a":"a","aa":"aa","i":"i","ee":"ee","u":"u","oo":"oo","ri":"ri",
    "e":"e","ai":"ai","o":"o","au":"au","am":"am","ah":"ah",
}

_DEV = {
    "\u0905":"a","\u0906":"aa","\u0907":"i","\u0908":"ee","\u0909":"u","\u090a":"oo","\u090b":"ri",
    "\u090f":"e","\u0910":"ai","\u0913":"o","\u0914":"au","\u0902":"n","\u0903":"h",
    "\u0915":"k","\u0916":"kh","\u0917":"g","\u0918":"gh","\u0919":"ng",
    "\u091a":"ch","\u091b":"chh","\u091c":"j","\u091d":"jh","\u091e":"ny",
    "\u091f":"t","\u0920":"th","\u0921":"d","\u0922":"dh","\u0923":"n",
    "\u0924":"t","\u0925":"th","\u0926":"d","\u0927":"dh","\u0928":"n",
    "\u092a":"p","\u092b":"ph","\u092c":"b","\u092d":"bh","\u092e":"m",
    "\u092f":"y","\u0930":"r","\u0932":"l","\u0935":"v","\u0936":"sh","\u0937":"sh","\u0938":"s","\u0939":"h",
    "\u093e":"a","\u093f":"i","\u0940":"ee","\u0941":"u","\u0942":"oo","\u0943":"ri",
    "\u0947":"e","\u0948":"ai","\u094b":"o","\u094c":"au","\u0902":"n","\u0901":"n",
    "\u094d":"","\u093c":"","\u0964":".","\u0965":".",
}


def translit(text):
    return "".join(_DEV.get(c, c) for c in str(text or ""))


NAME_ABBREVIATIONS = {
    "pvt":"private","ltd":"limited","corp":"corporation","inc":"incorporated",
    "llc":"limited liability company","co":"company","intl":"international",
    "mfg":"manufacturing","tech":"technologies","dept":"department",
    "univ":"university","assoc":"association","grp":"group",
    "svc":"services","svcs":"services","ent":"enterprises",
    "mgmt":"management","soln":"solutions","solns":"solutions",
    "llp":"limited liability partnership","lp":"limited partnership",
    "sarl":"societe a responsabilite limitee","sas":"societe par actions simplifiee",
    "sa":"societe anonyme","cie":"compagnie","ste":"societe",
}
ABBR_REGEX = re.compile(r"\b(" + "|".join(map(re.escape, NAME_ABBREVIATIONS)) + r")\b")

LEGAL_STOPWORDS = {
    "pvt","private","ltd","limited","corp","corporation","inc","incorporated",
    "llc","co","company","enterprises","services","technologies","group",
    "solutions","management","international","societe","sarl","sas","sasu",
    "and","the","for","with","india","france","states","united","llp","lp",
    "limited","liability","partnership","sa","cie","ste","an","of","by",
}

ADDR_STOPWORDS = {
    "street","road","avenue","lane","floor","block","near","opposite",
    "building","house","plot","sector","post","dist","nagar","suite",
    "apartment","apt","unit","drive","boulevard","court","place",
}


def norm_name(text):
    if not text:
        return ""
    t = translit(text).lower()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = ABBR_REGEX.sub(lambda m: NAME_ABBREVIATIONS[m.group(0)], t)
    return re.sub(r"[^a-z0-9 ]", "", t)


def norm_addr(text):
    if not text:
        return ""
    t = translit(text).lower()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_domain_name(text):
    if not text:
        return ""
    t = str(text).lower().strip()
    t = re.sub(r'\.(com|net|org|in|co|io|info|biz|us|fr|uk|de|au)(\.\w+)?$', '', t)
    t = re.sub(r'^(www\d?|ww\d)\.', '', t)
    t = re.sub(r'^(http[s]?://)?(www\.)?', '', t)
    return re.sub(r'[^a-z0-9]', '', t)


def get_name_tokens(norm_text, min_len=4):
    return [w for w in norm_text.split() if len(w) >= min_len and w not in LEGAL_STOPWORDS]


def get_addr_tokens(norm_text, min_len=4):
    return [w for w in norm_text.split() if len(w) >= min_len and w not in ADDR_STOPWORDS]


def extract_numbers(text):
    return re.findall(r"\d+", str(text or ""))


def text_for_tfidf(norm_name, norm_addr=""):
    dn = re.sub(r"[^a-z0-9]", "", norm_name)
    return (norm_name + " " + dn + " " + norm_addr).strip()


# =================== Features ===================

def compute_features(s1n, s1a, s1_nums, cn, ca, c_nums):
    return [
        ratio(s1n, cn) / 100.0,
        token_set_ratio(s1n, cn) / 100.0,
        partial_ratio(s1n, cn) / 100.0,
        WRatio(s1n, cn) / 100.0,
        ratio(s1a, ca) / 100.0,
        token_set_ratio(s1a, ca) / 100.0,
        partial_ratio(s1a, ca) / 100.0,
        float(len(set(s1_nums) & set(c_nums))),
        float(len(s1n)), float(len(cn)),
        float(len(s1a)), float(len(ca)),
    ]


# =================== Training ===================

def train_model():
    batches = sorted(glob.glob(str(ARTIFACTS_DIR / "training_batches" / "batch_*.csv")))
    if not batches:
        print("[WARN] No training batches found.")
        return None, None

    print(f"Training SGDClassifier on {len(batches)} batches...")
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-5,
                          class_weight={0: 1, 1: 3}, random_state=42)
    scaler = StandardScaler()

    df0 = pd.read_csv(batches[0])
    fcols = [c for c in df0.columns if c not in ["s1_id", "candidate_id", "label"]]
    scaler.fit(df0[fcols].values)

    for bpath in batches:
        df = pd.read_csv(bpath)
        if df.empty:
            continue
        model.partial_fit(scaler.transform(df[fcols].values), df["label"].values,
                          classes=np.array([0, 1]))

    print("  Model trained.")
    return model, scaler


def make_scorer(model, scaler):
    if model is None:
        return lambda feat: 0.5
    w = model.coef_[0]
    b = model.intercept_[0]
    m_ = scaler.mean_
    s_ = scaler.scale_

    def score(feat_vec):
        norm_f = (np.array(feat_vec, dtype=np.float64) - m_) / s_
        z = b + w.dot(norm_f)
        return 1.0 / (1.0 + math.exp(-z))
    return score


# =================== Index ===================

class CandidateIndex:
    def __init__(self):
        self.token_idx = defaultdict(list)
        self.domain_idx = defaultdict(list)
        self.addr_idx = defaultdict(list)
        self.records = {}  # eid -> (norm_n, norm_a, nums)

    def add(self, eid, name, addr):
        nn = norm_name(name)
        na = norm_addr(addr)
        nums = extract_numbers(addr)
        self.records[eid] = (nn, na, nums)

        for tok in get_name_tokens(nn, min_len=4)[:6]:
            self.token_idx[tok].append(eid)

        raw_lower = str(name or "").lower()
        if "." in raw_lower or re.match(r"^[a-z0-9]+\.(com|net|org|in)", raw_lower):
            dom = extract_domain_name(raw_lower)
        else:
            dom = re.sub(r"[^a-z0-9]", "", nn)
        if len(dom) >= 5:
            self.domain_idx[dom].append(eid)

        a_toks = get_addr_tokens(na, min_len=4)
        if nums and a_toks:
            self.addr_idx[nums[0] + ":" + a_toks[0]].append(eid)

    def lookup(self, name, addr):
        nn = norm_name(name)
        na = norm_addr(addr)
        nums = extract_numbers(addr)
        tokens = get_name_tokens(nn, min_len=4)
        a_toks = get_addr_tokens(na, min_len=4)

        seen = set()

        for tok in tokens[:6]:
            bucket = self.token_idx.get(tok, [])
            take = bucket if len(bucket) <= 50 else bucket[:TOKEN_TOP_K]
            seen.update(take)

        raw_lower = str(name or "").lower()
        if "." in raw_lower:
            dom = extract_domain_name(raw_lower)
        else:
            dom = re.sub(r"[^a-z0-9]", "", nn)

        if len(dom) >= 5:
            seen.update(self.domain_idx.get(dom, [])[:15])

        if nums and a_toks:
            seen.update(self.addr_idx.get(nums[0] + ":" + a_toks[0], [])[:10])

        return list(seen), nn, na, nums


# =================== TF-IDF ===================

def build_tfidf_index(index: CandidateIndex):
    print("  Fitting TF-IDF vectorizer on candidate corpus...")
    cand_eids = list(index.records.keys())
    cand_texts = [text_for_tfidf(index.records[e][0], index.records[e][1]) for e in cand_eids]

    vect = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 4),
        min_df=2, max_features=300_000, sublinear_tf=True,
    )
    cand_matrix = vect.fit_transform(cand_texts)
    print(f"  TF-IDF matrix: {cand_matrix.shape}")
    return vect, cand_matrix, cand_eids


def tfidf_lookup_batch(s1_batch, vect, cand_matrix, cand_eids, top_k=TFIDF_TOP_K):
    s1_texts = [text_for_tfidf(norm_name(str(r.get("business_name","") or "")),
                               norm_addr(str(r.get("business_address","") or "")))
                for r in s1_batch]
    s1_mat = vect.transform(s1_texts)
    sims = (s1_mat @ cand_matrix.T).toarray()

    results = {}
    for j, row in enumerate(s1_batch):
        s1_id = row["entity_id"]
        top_idx = np.argpartition(sims[j], -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(sims[j][top_idx])[::-1]]
        results[s1_id] = [cand_eids[i] for i in top_idx if sims[j, i] > 0.05]
    return results


# =================== Main ===================

def main():
    print("=" * 75)
    print("Amazon ML Challenge 2026 — Final Submission Builder v2")
    print("=" * 75)
    t_start = time.time()

    # 1. Train
    model, scaler = train_model()
    scorer = make_scorer(model, scaler)

    # 2. Build candidate index
    print("\nBuilding candidate index from Source2 + Source3...")
    index = CandidateIndex()
    total_cands = 0
    for fname in ("test_source2.tsv", "test_source3.tsv"):
        fpath = TEST_DIR / fname
        if not fpath.exists():
            print(f"  [SKIP] {fname} not found")
            continue
        print(f"  Indexing {fname}...")
        t0 = time.time()
        with open(fpath, encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 1:
                    continue
                eid  = p[0].strip()
                name = p[1].strip() if len(p) > 1 else ""
                addr = p[2].strip() if len(p) > 2 else ""
                index.add(eid, name, addr)
                total_cands += 1
        print(f"    {fname}: {time.time()-t0:.1f}s")

    print(f"Total candidates indexed: {total_cands:,}")

    # 3. Build TF-IDF
    print("\nBuilding TF-IDF index...")
    t0 = time.time()
    vect, cand_matrix, cand_eids = build_tfidf_index(index)
    print(f"  TF-IDF built in {time.time()-t0:.1f}s")

    # 4. Load S1
    print("\nLoading test_source1.tsv...")
    s1_df = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t",
                        encoding="utf-8", low_memory=False)
    print(f"  Source1 entities: {len(s1_df):,}")
    s1_records = s1_df.to_dict("records")

    # 5. Inference in batches
    print(f"\nRunning inference (threshold={MATCH_THRESHOLD})...")
    BATCH = 500
    matching = {}
    candidate_pairs = {}
    n_batches = math.ceil(len(s1_records) / BATCH)
    t0 = time.time()

    for bi in range(n_batches):
        batch = s1_records[bi * BATCH:(bi + 1) * BATCH]

        # TF-IDF candidates for this batch
        tf_cands = tfidf_lookup_batch(batch, vect, cand_matrix, cand_eids)

        for row in batch:
            s1_id = row["entity_id"]
            name  = str(row.get("business_name", "") or "")
            addr  = str(row.get("business_address", "") or "")

            idx_c, nn, na, nums = index.lookup(name, addr)
            tfc = tf_cands.get(s1_id, [])
            all_c = list(dict.fromkeys(idx_c + tfc))
            candidate_pairs[s1_id] = set(all_c)

            matched = set()
            for c_eid in all_c:
                rec = index.records.get(c_eid)
                if rec is None:
                    continue
                cn, ca, cnums = rec
                feat = compute_features(nn, na, nums, cn, ca, cnums)
                if scorer(feat) >= MATCH_THRESHOLD:
                    matched.add(c_eid)

            matching[s1_id] = matched

        if (bi + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = ((bi + 1) * BATCH) / elapsed
            eta = (n_batches - bi - 1) * BATCH / rate / 60
            print(f"  Batch {bi+1}/{n_batches}  "
                  f"rate={rate:.0f}/s  ETA={eta:.1f}min")

    elapsed = time.time() - t0
    total_matches = sum(len(v) for v in matching.values())
    print(f"  Inference done in {elapsed:.1f}s")
    print(f"  Total predicted matches: {total_matches:,}")

    # 6. Write outputs
    print("\nWriting output files...")
    mr_path = OUT_DIR / "matching_results.tsv"
    cp_path = OUT_DIR / "candidate_pairs.tsv"

    with open(mr_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, matches in matching.items():
            f.write(s1_id + "\t" + ",".join(sorted(matches)) + "\n")

    with open(cp_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cands in candidate_pairs.items():
            f.write(s1_id + "\t" + ",".join(sorted(cands)) + "\n")

    # 7. Validate
    print("\nValidating submission...")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(RESOURCE / "utils" / "validate_submission.py")],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    print(result.stdout[-2000:])
    if result.returncode == 0:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED:", result.stderr[-500:])

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
