# Entity Resolution Model Explanation

This document explains the step-by-step process used to build the machine learning model for the Amazon ML Challenge, written in simple language.

## The Goal
We have three large datasets (Source 1, Source 2, and Source 3) that contain lists of businesses. The goal is to figure out which businesses in Source 1 are the exact same as businesses in Source 2 and Source 3, even if their names or addresses are written slightly differently.

---

## Step-by-Step Process

### 1. Multilingual Cleaning & Normalization

Businesses across countries are written in different languages, scripts, and formats. We handle three main cases:

#### 1a. Hindi / Devanagari Script → Latin
India is one of the countries in the dataset. Some names and addresses may contain Devanagari (Hindi) script. Since our fuzzy matching engine only works on Latin characters, we **phonetically transliterate** Devanagari to Latin before any processing:
- `टाटा मोटर्स` → `tata motrs` (phonetic approximation)
- `एम जी मार्ग` → `em jee marg`

This is done character-by-character using a lookup table (`DEVANAGARI_MAP`) — no external library needed.

#### 1b. French / Latin Accented Characters → ASCII
France is also in the dataset. Characters like `é`, `è`, `ê`, `ô`, `ç` are common in French business names:
- `École primaire` → `ecole primaire`
- `Société SARL` → `societe sarl`

This is handled by Unicode NFKD normalization + combining character stripping — a standard approach that works for French, Spanish, German, and other European scripts.

#### 1c. Corrupted Characters → Space
Some records may contain garbled/corrupted bytes (the `<replacement character>`). We replace these with spaces so they don't interfere with matching.

#### 1d. Abbreviation Expansion
After character normalization, we expand standard abbreviations using word-boundary regex:

| Type | Abbreviation → Full Form |
|------|--------------------------|
| **English business** | `pvt` → `private`, `ltd` → `limited`, `inc` → `incorporated`, `llc` → `limited liability company`, `corp` → `corporation` |
| **French legal entities** | `sarl` → `societe a responsabilite limitee`, `sasu` → `societe par actions simplifiee unipersonnelle`, `sa` → `societe anonyme`, `ei` → `entreprise individuelle` |
| **English addresses** | `rd` → `road`, `st` → `street`, `ave` → `avenue`, `apt` → `apartment`, `blvd` → `boulevard` |
| **French addresses** | `bd` → `boulevard`, `av` → `avenue`, `r` → `rue`, `etg` → `etage` |
| **Indian addresses** | `marg` → `road`, `chowk` → `square`, `bazar` → `market`, `stn` → `station`, `sec` → `sector` |

This ensures "Apple Inc." and "Apple Incorporated" normalize to the **exact same string**, so fuzzy matching sees a 100% match.

---

### 2. Full-Source Indexing Without Loading Sources into RAM

The supplied training files contain 2,206,821 Source-1 rows, and the test files contain 1,732,544 Source-1 rows plus more than 9.9 million Source-2/3 rows. The full Source-2/3 files are read in 20,000-row chunks. Their normalized records and blocking postings are stored in a temporary SQLite database on disk, not in large Python dictionaries in RAM.

The ground truth is also read in chunks and indexed on disk. Training processes every Source-1 row except a deterministic 1% holdout used to select the F0.5 threshold. Each training batch is discarded after `SGDClassifier.partial_fit()` updates the model. The test index is built after training, and the temporary database is removed when inference completes.

---

### 3. Candidate Generation via Blocking — *The Key Computation Saver*

#### Why Naive Comparison Is Impossible

Source 1 has ~2.2 million rows. Source 2 has ~5 million rows and Source 3 has ~5.3 million rows.

If we compared **every** S1 record against **every** S2 and S3 record, we'd need:
> 2.2M × (5M + 5.3M) = **~22.66 trillion pairs**

Even at 1 microsecond per pair, this would take **261 days**. It is completely infeasible.

#### How Blocking Works (The Index Analogy)

Think of it like the **index at the back of a textbook**. Instead of reading every page to find a topic, you look up the word in the index and it tells you exactly which pages to check.

We build two disk-backed inverted indexes (before any S1 processing begins), covering every row in both Source-2 and Source-3:

**Index 1 — Name Token Index:**
```
"tata"   → [S2-row-4, S2-row-17, S3-row-902, ...]
"motors" → [S2-row-4, S3-row-11, ...]
"apple"  → [S2-row-99, S2-row-231, S3-row-44, ...]
```
For each S2/S3 record, we extract its **distinctive name words** (≥3 characters, excluding generic legal words like `limited`, `corporation`, `societe`) and add its entity ID to the SQLite index under each word.

**Index 2 — Address Number Index:**
```
"123" → [S2-row-4, S3-row-7, S3-row-801, ...]
"456" → [S2-row-19, S3-row-45, ...]
```
Street/unit numbers are powerful matching signals — if two records share the number `123`, they're much more likely to be the same address. These postings are stored in the same disk-backed index with a separate token prefix.

#### Candidate Retrieval for One S1 Record

When we process a single S1 record (e.g., *"Tata Motors Ltd, 123 MG Road, India"*):

1. Extract its name tokens: `["tata", "motors"]`
2. Look them up in the name index → get a **small set of candidate positions** from S2 and S3
3. Extract its address numbers: `["123"]`
4. Look them up in the address number index → add more candidates
5. **Filter by country** — only keep candidates from the same country (India in this case)
6. **Cap at 500 candidates** per source to prevent edge-case explosions

**Result:** Instead of comparing against 10.3M records, we compare against maybe **50–500 records** — a **20,000× speedup**.

#### Why We Filter Out Legal Stopwords from the Index

Words like `limited`, `corporation`, `societe`, `incorporated` appear in thousands of business names. If we indexed them, looking up "Apple Limited" would return hundreds of thousands of candidates (every "Limited" company in the world), flooding the candidate set with false positives and overwhelming the cap.

By stripping these words from the token index (but keeping them in the normalized strings for fuzzy matching), we ensure the index only retrieves records sharing **meaningful business identity words**.

#### The Computation Saving in Numbers

| Approach | Pairs Evaluated | Time Estimate |
|----------|----------------|---------------|
| Naive full cross-product | ~22.66 trillion | 261 days |
| **Blocking (our approach)** | ~200–500 per S1 record | **Minutes to hours** |

---

### 4. Feature Engineering (Giving the Model Clues)

Once we have our short list of candidates, we compute 12 similarity features for each (S1, candidate) pair using **RapidFuzz** — a fast C-backed fuzzy matching library:

| Feature | What It Measures |
|---------|-----------------|
| `name_ratio` | Edit-distance similarity of names (0–1) |
| `name_token_set_ratio` | Token-level intersection — handles word reordering |
| `name_partial_ratio` | Best substring match — handles truncated names |
| `address_ratio` | Edit-distance similarity of addresses |
| `address_token_set_ratio` | Token-level intersection for addresses |
| `address_partial_ratio` | Best substring match for addresses |
| `country_match` | 1 if same country, 0 otherwise |
| `num_overlap_count` | Number of address digits shared |
| `name_len_s1`, `name_len_cand` | Length signals (short vs long names) |
| `addr_len_s1`, `addr_len_cand` | Address length signals |

RapidFuzz is called **only on candidate pairs** — never on full cross-products.

---

### 5. Training the Model in Batches

The production runner uses `SGDClassifier` with logistic loss and `partial_fit()`. It processes all 2,206,821 labeled Source-1 records in 500-row batches, holding out a deterministic 1% of IDs for threshold selection. For each training batch it generates blocked pairs, samples negatives, updates the model, then releases the batch. The record indexes and ground-truth lookup remain on disk in SQLite, so Source-2/3 data and all training pairs are never accumulated in RAM.

---

### 6. Setting the Rules (Thresholding)

The model outputs a probability score (0–1) for each candidate pair. We sweep all thresholds from 0.10 to 0.90 and pick the one that maximizes **F0.5 score** (which rewards precision more than recall, per competition rules).

---

### 7. Making Final Predictions

The trained model indexes all test Source-2/3 records in chunks, then processes all 1,732,544 test Source-1 rows in batches of 500. For each batch:
1. Generate candidates (using the pre-built index)
2. Compute features for each candidate pair
3. Predict probabilities
4. Accept pairs above `BEST_THRESHOLD`
5. Write results to TSV files incrementally
6. Free memory and move to the next batch

At no point are the full test Source-2/3 files or all candidate pairs held in memory simultaneously. The output files are written to temporary files first and published only after the full inference pass completes.

### 8. Run and Validate

From the project root, run `python New_Model_script.py` (or run `New_Model.ipynb`). The pipeline writes `student_resource/output/matching_results.tsv` and `student_resource/output/candidate_pairs.tsv`, mirrors them to the root `output/` folder, and runs `student_resource/utils/validate_submission.py` against the complete test Source-1 list before reporting success. The temporary SQLite index is deleted after each phase.
