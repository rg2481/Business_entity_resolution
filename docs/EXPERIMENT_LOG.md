# Experiment log: from 0.9636 to 0.9822

This log condenses about 3 days of experiments and 20+ leaderboard uploads into the steps that mattered.
Scores are **public leaderboard macro F0.5** unless marked *local* (held-out validation, see "Validation" in the README).

## Leaderboard timeline

| # | Submission | What changed | Public F0.5 |
|---|---|---|---|
| 1 | B01 baseline | GPU TF-IDF retrieval, then LightGBM on 61 string-similarity features | 0.963571 |
| 2 | C04 | +31 perturbation-aware features, 8x training data (local 0.9714 → 0.9844) | 0.970 |
| 3 | C04 + France probe | B01's French rows only, to isolate France | 0.969 |
| 4 | C04 + US near-number rule | drop same-name US matches whose house number is 1–9 away | 0.972505 |
| 5 | Teammate model | an independent LightGBM pipeline | 0.971925 |
| 6 | Blend s1 | drop C04 matches in its 0.76–0.90 band that the teammate model rejects | 0.976609 |
| 7 | Blend final | + house-mismatch removals (France/US), + India additions | 0.978885 |
| 8 | Blend v2 | + pattern-checked US/India additions | 0.979052 |
| 9 | **C28 pure (own pipeline only)** | fine-tuned multilingual cross-encoder + LightGBM stack | **0.980736** |
| 10 | C28 combo (model 1) | cross-encoder vetoes/rescues on top of blend v2 | 0.981184 |
| 11 | C28 combo (model 2) | cross-encoder trained on 2x data | 0.982124 |
| 12 | Score fusion (C31) | small fusion refinement | 0.982150 |
| 13 | France self-training (C33) | one pseudo-label round on French test pairs | **0.982171 (final team best)** |

This repository reproduces the **own-pipeline** route (row 9, with the model-2 upgrade) end to end. Rows 5–8 and 10–13
also used a teammate's separate pipeline, whose code is not part of this repository.

## What moved the score, and why

1. **Record-centric retrieval (B01).**
   - The ground truth is a strict partition: each S2/S3 record has at most one owner. So every record searches all
     S1 businesses in its country, using the union of the top 10 by name, by address and by name+address.
   - 98.7–99.0% of true matches are retrieved, while the candidate set shrinks by 99.996%.
2. **Features that describe *how* a record was perturbed (C04, +0.0064).**
   - EDA showed the generator's noise families: dropped or added legal words, web handles, typos, Indic
     transliterations, shifted house numbers.
   - Word-level name relations, role-aware house/unit numbers, one-digit substitutions and "twin" counts across
     records took local F0.5 from 0.971 to 0.984.
3. **Diagnosing a train/test shift with controlled uploads (+0.0025).**
   - Only half of C04's local gain reached the leaderboard.
   - An upload that changed only French rows proved France was not the cause.
   - A label-free comparison then found far more **sibling decoys** in US test data: same name and street, house
     number shifted by a few. A narrow rule rejecting them gained +0.0025.
4. **A second, differently built model (blend, +0.0065).**
   - A teammate's simpler model scored nearly the same (0.9719) while disagreeing on 17% of S1 rows.
   - Its rejections were highly informative exactly where C04 was only moderately confident. On test, C04 was
     overconfident on house-number mismatches.
   - A two-parameter truth model fitted to leaderboard results priced each blend before upload; it predicted the
     first blend within 0.0001.
5. **A transformer reading both raw records (C28, +0.0031 over the blend).**
   - A multilingual cross-encoder (`intfloat/multilingual-e5-small`, MIT, 118M params) was fine-tuned on raw
     `name | address` pairs and combined with the LightGBM score in a logistic stack.
   - *Local:* 0.98442 → 0.98821 (25% of the data) → 0.98881 (50%).
   - On its own, with no teammate input, it reached 0.9807 on the leaderboard. It independently rejects 62% of the
     decoys the blend had found. Its local gains transferred to the leaderboard, unlike earlier feature tweaks.

## Things that did not work (and were not submitted)

- **XGBoost / bagging / equal-weight ensembles of the same features:** −0.001 to +0.0003 locally.
- **Country-weighted training and country-weighted thresholds:** no gain.
- **A global "house number must match" rule:** −0.01 to −0.03 locally. The extracted "house number" is often a plot,
  block or road number, and matched pairs with a different number are still 95–99% true.
- **Re-ranking C04's top-3 candidates with the transformer:** about +0.0000. C04 almost always ranks the right S1 first.
- **French "same house number" additions:** they sit on different streets, so they are different businesses.
- **Extra phonetic transliteration for Indic text:** native-script records were already matched *more* accurately
  (recall 0.976–0.996) than Latin ones (0.966). All records with dictionary-unknown words were distractors.

## Lessons

- Validate every idea on labeled held-out data before spending a scarce upload, and **read raw samples before every rule**.
- When local and public scores diverge, design uploads that isolate one question: one country, or one rule.
- A second model is valuable in proportion to how *different* its evidence is. B01's vote added nothing, because
  C04 already contained its features; a text model reading raw strings added a lot.
