# Methodology and Reproducibility Notes

This document describes how the analysis in `notebooks/telecom_churn_prediction_xgboost.ipynb` was designed, what each reported number means, and where its limits lie. All figures below are taken from the executed notebook.

## 1. Data and target

| Item | Detail |
|---|---|
| `Client.csv` | 100,000 rows × 50 columns |
| `Record.csv` | 100,000 rows × 51 columns |
| Merge key | `Customer_ID`: unique, non-missing, 100,000 matches in both tables, no orphans |
| Merged table | 100,000 × 100 (79 numeric + 21 categorical, including ID and label) |
| Target | `churn` (1 = churned between 31 and 60 days after the observation date, per source documentation) |
| Class balance | 49,562 churners (49.56%), 50,438 retained (50.44%) |
| Predictors | 98 original features (77 numeric, 21 categorical) after excluding `Customer_ID` and `churn` |

## 2. Data integrity checks

- One-to-one merge with pandas `validate="one_to_one"`, plus assertions on row count, ID uniqueness, and ID completeness.
- No fully duplicated rows, no constant or empty columns, no infinite values in training data, no entirely missing numeric columns.
- 342,969 missing cells in total. Six columns exceed 30% missing: `numbcars` (49.37%), `dwllsize` (38.31%), `HHstatin` (37.92%), `ownrent` (33.71%), `dwlltype` (31.91%), `lor` (30.19%).
- Missingness is mildly associated with churn: churn is 1.33 to 2.57 percentage points higher when these fields are missing. This motivates explicit missing indicators but is not evidence of a causal mechanism.

## 3. Exploratory analysis

- Median comparison of churned versus retained customers (e.g. `eqpdays` 366 vs 310; `mou_Mean` 329.75 vs 380.50).
- Quintile churn rates for `eqpdays`, `mou_Mean`, and `ovrrev_Mean`. Device-age churn is non-monotonic: 41.5%, 37.4%, 59.5%, 53.0%, 56.4% across quintiles.
- Combined 3 × 3 segmentation (device age ≤ 300 / 301 to 395 / > 395 days × low / medium / high `mou_Mean`, using cut-points of 265.5 and 821.5 minutes, which are quintile boundaries). The highest-churn segment is 301 to 395 days with low usage: 61.84% churn, 7,894 customers.
- The segmentation covers 99,642 customers; 358 are excluded for missing `eqpdays` (1) or `mou_Mean` (357).
- Cut-points were chosen by looking at the full dataset, so segment rates are descriptive and exploratory, not out-of-sample estimates.

## 4. Validation design

- Stratified random split on the label: **70,000 train / 15,000 validation / 15,000 test** (`random_state=42`), implemented as a 70/30 split followed by a 50/50 split of the remainder.
- Churn rate is 49.56% to 49.57% in every split.
- Assertions confirm that the splits do not overlap, cover all rows, and exclude the label and ID from predictors.
- All preprocessing (imputers, missing indicators, one-hot encoders, and the Logistic Regression scaler) is fit on the training split only and applied unchanged to validation and test.
- Model selection used validation metrics only. The test set was scored once for final predictive reporting; those same test predictions were then reused, without refitting or reselection, for the retrospective targeting and financial analysis.

**Not done:** time-based (chronological) backtesting, cross-validation, nested validation, or repeated random splits. Uncertainty intervals for the reported metrics were not computed.

## 5. Preprocessing

| Feature group | Steps |
|---|---|
| Numeric (77) | Median imputation with missing-value indicator columns (indicators are created for columns with missing values in training) |
| Categorical (21) | Constant imputation with `"Missing"`, then one-hot encoding with `handle_unknown="ignore"` |
| Output | 272 columns (sparse where beneficial) for train, validation, and test; zero non-finite values |

## 6. Models

| Model | Configuration |
|---|---|
| Logistic Regression (baseline) | `StandardScaler(with_mean=False)` + `lbfgs`, `max_iter=2000` (converged in 255 iterations) |
| XGBoost, initial | 500 trees, depth 5, learning rate 0.05, subsample 0.8, column subsample 0.8, no early stopping |
| **XGBoost, final** | Up to 1,500 trees, binary logistic objective, histogram trees, CUDA device 0, max depth 6, learning rate 0.03, subsample 0.8, column subsample 0.8, minimum child weight 3, L2 regularization (`reg_lambda`) 2, early stopping patience 75 on validation ROC-AUC |

Final model: best iteration 959 (zero-based, i.e. 960 trees retained), 1,035 trees built before stopping, validation ROC-AUC 0.7032. Libraries: XGBoost 3.4.1, Python 3.13, Kaggle environment with Tesla T4 GPUs (one device used).

**On "tuning":** the final model is a single manually adjusted configuration compared against the initial one, not the output of a systematic search. Its validation gain over the initial model is small (0.7032 vs 0.7013 ROC-AUC) and should not be over-interpreted. Because the early-stopping round was chosen on the validation set, validation scores are slightly optimistic; this is why the untouched test set is used for final reporting.

**Reproducibility:** seeds are fixed, but GPU training is not guaranteed to be bit-identical across hardware, drivers, or library versions. Treat reported scores as reproducible to roughly the third decimal place, not as universal constants.

## 7. Classification and probability metrics

| Model | Dataset | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Validation | 0.6328 | 0.6081 | 0.5971 | 0.5942 | 0.5906 | 0.5924 |
| XGBoost (initial) | Validation | 0.7013 | 0.6914 | 0.6419 | 0.6362 | 0.6483 | 0.6422 |
| XGBoost (final) | Validation | 0.7032 | 0.6923 | 0.6419 | 0.6372 | 0.6443 | 0.6407 |
| **XGBoost (final)** | **Test** | **0.6945** | **0.6816** | **0.6401** | **0.6340** | **0.6476** | **0.6407** |

- Accuracy, precision, recall, and F1 use a 0.5 probability threshold.
- Test Brier score 0.2206. On validation, Brier is 0.2181 and log loss is 0.6248, against a constant-prevalence Brier of 0.2500.
- Test confusion matrix (rows actual retained/churn, columns predicted retained/churn): `[[4787, 2779], [2620, 4814]]`.
- Validation calibration (10 quantile bins) is close to the diagonal; the largest gap is 2.88 percentage points (bin 5). Calibration applies to this balanced sample and would not carry over to a population with a different churn rate.

**Interpretation of PR-AUC:** with about 50% prevalence the no-skill PR-AUC is about 0.50, so 0.6816 is a moderate improvement, not a high score.

## 8. Risk-ranked targeting

Customers in the test set are sorted by predicted churn probability; for the top *k*% the notebook reports captured churners, group churn rate, recall, and lift (group churn rate ÷ overall churn rate).

| Top share | Customers | Churners captured | Churn rate | Recall | Lift |
|---:|---:|---:|---:|---:|---:|
| 5% | 750 | 638 | 85.07% | 8.58% | 1.716 |
| 10% | 1,500 | 1,193 | 79.53% | 16.05% | 1.605 |
| 20% | 3,000 | 2,191 | 73.03% | 29.47% | 1.474 |
| 30% | 4,500 | 3,102 | 68.93% | 41.73% | 1.391 |

Because the sample is about 50% churn, lift cannot exceed roughly 2.02×. Absolute precision and lift in an imbalanced production population would be different.

## 9. Revenue proxy

`Monthly_Revenue_Proxy` equals `rev_Mean`, except for the **52** test-set rows where `rev_Mean` is missing (51) or negative (1); for these, `avgrev` is substituted. This is a scenario assumption and does not repair the underlying accounting records. Two customers retain a proxy of exactly zero. The proxy is revenue, not profit. Currency is unspecified; all monetary values are in "units".

Test-set revenue proxy by risk group:

| Top share | Customers | Total monthly revenue | Churner monthly revenue |
|---:|---:|---:|---:|
| 10% | 1,500 | 79,060.96 | 62,363.07 |
| 20% | 3,000 | 168,503.17 | 122,198.84 |
| 30% | 4,500 | 255,988.45 | 175,389.97 |
| 100% | 15,000 | 878,417.07 | 436,031.83 |

## 10. Value-aware ranking

An alternative ranking score is `Revenue_At_Risk = P(churn) × Monthly_Revenue_Proxy`.

| Strategy | Budget | Churners | Churn rate | Churner monthly revenue |
|---|---:|---:|---:|---:|
| Risk only | 10% (1,500) | 1,193 | 79.53% | 62,363.07 |
| Value aware | 10% (1,500) | 906 | 60.40% | 128,284.33 |
| Risk only | 20% (3,000) | 2,191 | 73.03% | 122,198.84 |
| Value aware | 20% (3,000) | 1,798 | 59.93% | 199,532.58 |

At the 20% budget, value-aware ranking has 63.3% higher retrospective churner revenue exposure and 393 fewer churners. Revenue exposure is not recoverable profit: it assumes nothing about who responds to treatment.

## 11. Economic model (illustrative)

**Assumptions (not company-provided):** M = 6 additional retained months, m = 60% contribution margin, contact cost c = 2, incentive cost i = 15 per contacted customer. Every targeted customer incurs both costs.

For N contacted customers and relative churn reduction *e*:

```text
Campaign cost        = N × (c + i)
Retained contribution = R_churn × e × M × m
Net benefit          = Retained contribution − Campaign cost
Break-even e*        = N × (c + i) / (R_churn × M × m)
```

where `R_churn` is the observed churner monthly revenue proxy in the targeted group.

For the top 20% (N = 3,000): cost = 3,000 × 17 = **51,000**; `R_churn` = 122,198.84; e* = 51,000 / (122,198.84 × 6 × 0.60) = **11.59%**.

| Target share | N | Cost | e* | Net benefit at e = 10% |
|---:|---:|---:|---:|---:|
| 10% | 1,500 | 25,500 | 11.36% | −3,049.30 |
| 20% | 3,000 | 51,000 | 11.59% | −7,008.42 |
| 30% | 4,500 | 76,500 | 12.12% | −13,359.61 |

Sensitivity for the top 20% (retention effect 5% to 30% × incentive cost 5 to 25) is saved to `financial_sensitivity.csv`. Break-even by incentive cost: 4.77% (5), 8.18% (10), 11.59% (15), 15.00% (20), 18.41% (25).

**Important simplifications:**
- The model uses *observed* churn labels only to describe the retrospective scenario. A real policy cannot know future outcomes and must target on predicted risk alone.
- The same relative effect is assumed for every contacted customer. Real responsiveness varies and is not the same as churn risk.
- Churn prevalence in the targeted group (73%) reflects a balanced sample. At lower real prevalence, fewer contacted customers would churn, `R_churn` would shrink, and break-even e* would rise.
- Fixed costs, discounting, cannibalization, customer lifetime beyond six months, and spillover effects are ignored.

## 12. Randomized pilot design

**Illustrative design:** 3,000 eligible participants randomly split into 1,500 treatment and 1,500 control; two-sided α = 0.05; 80% power; independent proportions (Cohen's *h* via `proportion_effectsize`, `NormalIndPower`). Minimum detectable effect by assumed baseline churn:

| Baseline churn | Treatment churn at MDE | Absolute reduction | Relative reduction |
|---:|---:|---:|---:|
| 10% | 7.15% | 2.85 pp | 28.55% |
| 20% | 16.07% | 3.93 pp | 19.64% |
| 30% | 25.42% | 4.58 pp | 15.25% |
| 50% | 44.89% | 5.11 pp | 10.21% |

At a 20% baseline, the break-even effect (≈ 11.6% relative, 2.3 pp) is well below the 3,000-customer MDE; detecting it would require roughly 4,500 customers per arm by a normal-approximation estimate (a rough hand calculation, not an output of the notebook; re-run the power analysis with real inputs). Actual sample size must be set from observed baseline churn, planned treatment, the minimum economically meaningful effect, and expected attrition.

**Recommended protocol elements:**
- Eligibility defined before randomization; random assignment before any contact.
- Pre-registered primary outcome (churn within a window matching the label definition, 31 to 60 days after observation), secondary outcomes (revenue, incremental contribution, offer take-up), and analysis plan (intention-to-treat).
- Stratified randomization by risk band; consider an additional arm that randomizes risk-based versus random selection.
- Guardrail metrics: complaints, opt-outs, contact fatigue, and outcomes across demographic groups.
- Decision rule: scale only if the confidence interval for incremental contribution per contacted customer supports commercial viability, not just a statistically significant churn reduction.

A pilot scenario table (`pilot_financial_scenarios.csv`) reproduces the 11.59% break-even using treated-customer costs only. Its expected-value figures (for example, a net benefit of +7,494 at a 15% relative reduction and −3,504 at 10%) are hypothetical and are not estimated effects.

## 13. Feature importance

Importance is XGBoost **total gain**, normalized to 100% across 272 processed features. Top features: `eqpdays` (8.60%), `months` (5.69%), `change_mou` (4.55%), `mou_Mean` (3.68%), `totmrc_Mean` (3.39%). Gain-based importance reflects use within the model, can be biased toward high-cardinality or continuous features, and is not causal. Correlated features split importance between them.

## 14. Limitations and threats to validity

- **Sampling:** the sample appears balanced for churn; prevalence, calibration, precision, and economics will not transfer directly to the full customer base.
- **Temporal validity:** random splitting can overstate performance relative to forward-in-time use. Feature timing relative to the observation date should be audited for leakage.
- **Single split:** no confidence intervals; with 15,000 test customers, small differences between models are within plausible sampling noise.
- **Limited tuning:** one manual configuration; no systematic hyperparameter search.
- **Proxy limits:** revenue proxies are not margin; `rev_Mean` data-quality issues were patched only for the financial scenarios, not the model inputs.
- **No causal evidence:** no treatment or response data exist. Prediction is not uplift.
- **Fairness and privacy:** inputs include demographic and household variables. These need review for lawful basis, consent, and discriminatory proxy effects before use in individualized targeting.

## 15. Output artifacts

Ten CSV tables are written by the notebook: `model_comparison`, `final_test_metrics`, `segment_analysis`, `feature_importance`, `test_targeting`, `targeting_strategy_comparison`, `financial_scenarios`, `financial_sensitivity`, `pilot_power`, and `pilot_financial_scenarios`.
