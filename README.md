# Telecom Customer Churn Prediction & Retention Analytics

**A reproducible, business-focused machine learning case study: leakage-safe validation, GPU-accelerated XGBoost, budget-constrained targeting, campaign break-even analysis, and a randomized-pilot design.**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Retrospective%20PoC-lightgrey)

**Author:** Sana Ur Rehman Arain
**Project type:** Data science · Binary classification · Customer analytics
**Status:** Completed retrospective proof of concept. No live retention experiment has been conducted.

---

## Project overview

A telecom provider wants to identify customers at risk of churn and prioritize commercially sensible retention outreach. This project merges customer-profile and usage records, builds a churn-ranking model, examines which customer segments churn most, and tests whether a limited-budget retention campaign could plausibly pay for itself under transparent, hypothetical assumptions.

**Business question:** Can a predictive churn model concentrate likely churners within a limited outreach budget, and under what assumptions might a targeted retention intervention be financially worthwhile?

**Short answer from this study:** The model ranks customers meaningfully better than chance (test ROC-AUC 0.6945), but modestly. Under the stated assumptions, the campaign needs to prevent roughly **11.6% of expected churn** among contacted customers just to break even. Whether outreach achieves that is unknown and can only be established by a randomized pilot.

## Key results

| Result | Finding |
|---|---:|
| Customers analyzed | 100,000 |
| Original model inputs | 98 (77 numeric, 21 categorical) |
| Model inputs after preprocessing | 272 |
| Final model | Tuned XGBoost (GPU) |
| Held-out test ROC-AUC | **0.6945** |
| Held-out test PR-AUC (average precision) | **0.6816** |
| Held-out test accuracy | **64.01%** |
| Top 20% risk targeting lift | **1.474×** vs. random selection |
| Share of observed churners captured by top 20% | **29.47%** (2,191 of 7,434) |
| Churner monthly revenue proxy, top 20%, risk only | **122,199** units |
| Churner monthly revenue proxy, top 20%, value aware | **199,533** units |
| Break-even relative churn reduction (illustrative) | **≈ 11.59%** |

> **Read this first.** These are *retrospective sample results*, not deployment outcomes. Currency is unspecified. The dataset is approximately balanced for churn (49.56%), so its prevalence must not be taken as a real-world company rate. In a balanced sample the maximum possible top-20% lift is about 2.0×, so 1.474× should be read against that ceiling, not against typical imbalanced-churn benchmarks.

## Data

| File | Rows × columns | Content |
|---|---|---|
| `Client.csv` | 100,000 × 50 | Customer profile, demographics, handset and billing history |
| `Record.csv` | 100,000 × 51 | Usage, quality-of-service and call-behaviour metrics, plus the `churn` label |

- **Merge:** one-to-one inner join on `Customer_ID` (unique and non-missing in both tables), giving 100,000 × 100.
- **Target:** binary `churn`, defined in the source documentation as churn between 31 and 60 days after an observation date (49,562 churners / 50,438 retained).
- **Data quality:** no duplicate rows, no constant columns, no infinite values. 342,969 missing cells overall; six columns exceed 30% missing (`numbcars` 49.4%, `dwllsize` 38.3%, `HHstatin` 37.9%, `ownrent` 33.7%, `dwlltype` 31.9%, `lor` 30.2%). Missingness is weakly informative (churn is 1.3 to 2.6 percentage points higher when these fields are missing), which is why missing indicators are retained.
- **Revenue fields:** in the test split `rev_Mean` has 51 missing and 1 negative value; these 52 rows use `avgrev` as a fallback in the financial analysis only.

## Methods

1. **Data validation:** verify two source tables, unique IDs, missingness, and target distribution; merge one-to-one.
2. **Exploratory analysis:** equipment age, usage, revenue, missingness, and combined customer segments.
3. **Splitting:** stratified 70% train (70,000) / 15% validation (15,000) / 15% test (15,000), `random_state=42`. `Customer_ID` and `churn` are excluded from predictors.
4. **Preprocessing (fit on training data only):** median imputation plus missing indicators for numeric features; constant `"Missing"` category plus one-hot encoding for categorical features.
5. **Modeling:** scaled Logistic Regression baseline; GPU XGBoost; a second, deeper and more regularized XGBoost configuration with validation-based early stopping.
6. **Evaluation:** ROC-AUC, PR-AUC, accuracy, precision, recall, F1, Brier score, calibration, and top-risk lift. The test set is scored once for final reporting.
7. **Business analysis:** risk-only versus `probability × monthly revenue proxy` ranking; campaign break-even and sensitivity; randomized-pilot power analysis.

Full detail is in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Model performance

**Validation comparison (15,000 customers)**

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6328 | 0.6081 | 0.5971 | 0.5942 | 0.5906 | 0.5924 |
| XGBoost (initial, 500 trees) | 0.7013 | 0.6914 | 0.6419 | 0.6362 | 0.6483 | 0.6422 |
| **XGBoost (tuned, early stopping)** | **0.7032** | **0.6923** | 0.6419 | 0.6372 | 0.6443 | 0.6407 |

**Final held-out test (tuned XGBoost):** ROC-AUC 0.6945 · PR-AUC 0.6816 · accuracy 0.6401 · precision 0.6340 · recall 0.6476 · F1 0.6407 · Brier 0.2206 (constant-prevalence reference ≈ 0.250 on validation).

Confusion matrix (rows actual, columns predicted; threshold 0.5):

|  | Predicted retained | Predicted churn |
|---|---:|---:|
| **Actual retained** | 4,787 | 2,779 |
| **Actual churn** | 2,620 | 4,814 |

The validation-to-test ROC-AUC drop (0.7032 → 0.6945) is small and consistent with mild selection on the validation set. Validation calibration is close to the diagonal across risk deciles, with the largest decile gap about 2.9 percentage points; this holds only for this balanced sample.

## Risk-ranked targeting (test set, 15,000 customers)

| Top risk share | Customers contacted | Churners captured | Churn rate in group | Recall | Lift |
|---:|---:|---:|---:|---:|---:|
| 5% | 750 | 638 | 85.07% | 8.58% | 1.716× |
| 10% | 1,500 | 1,193 | 79.53% | 16.05% | 1.605× |
| **20%** | **3,000** | **2,191** | **73.03%** | **29.47%** | **1.474×** |
| 30% | 4,500 | 3,102 | 68.93% | 41.73% | 1.391× |

At the 3,000-customer budget, random selection would capture about 1,487 churners.

## Risk-only vs. value-aware ranking

At the same 3,000-customer budget:

| Strategy | Observed churners | Churn rate in group | Total monthly revenue (proxy) | Churner monthly revenue (proxy) |
|---|---:|---:|---:|---:|
| Risk only | 2,191 | 73.03% | 168,503 | 122,199 |
| Value aware (`P(churn) × revenue`) | 1,798 | 59.93% | 346,856 | 199,533 |

Value-aware ranking exposes **63.3% more** historical churner revenue but reaches **393 fewer churners**. This is a coverage-versus-value trade-off. It does **not** show higher recoverable profit, because nothing is known about whether high-revenue customers respond to outreach.

## Campaign economics (illustrative, not observed)

Assumptions: 6 additional retained months, 60% contribution margin, 2 units contact cost and 15 units incentive cost per contacted customer, all targeted customers incur both costs.

| Target share | Customers | Campaign cost | Break-even relative churn reduction | Net benefit at 10% effect |
|---:|---:|---:|---:|---:|
| 10% | 1,500 | 25,500 | 11.36% | −3,049 |
| **20%** | **3,000** | **51,000** | **11.59%** | **−7,008** |
| 30% | 4,500 | 76,500 | 12.12% | −13,360 |

Break-even is highly sensitive to incentive cost (top 20% group):

| Incentive cost per customer | 5 | 10 | 15 | 20 | 25 |
|---|---:|---:|---:|---:|---:|
| Break-even relative reduction | 4.77% | 8.18% | 11.59% | 15.00% | 18.41% |

With a 15-unit incentive, a 10% effect loses 7,008 units while a 15% effect gains 14,987 units. Because the sample is balanced, real-world churn among contacted customers would likely be lower than the 73% seen here, which would **raise** the true break-even threshold.

## Randomized pilot design

A 3,000-customer pilot (1,500 treatment / 1,500 control), two-sided α = 0.05, 80% power, independent proportions:

| Assumed baseline churn | Minimum detectable reduction | Relative reduction |
|---:|---:|---:|
| 10% | 2.85 pp | 28.6% |
| 20% | 3.93 pp | 19.6% |
| 30% | 4.58 pp | 15.2% |
| 50% | 5.11 pp | 10.2% |

If real baseline churn in the pilot population is low, a 3,000-customer pilot cannot reliably detect an effect as small as the break-even threshold, and a larger sample would be needed. The pilot must pre-declare its primary outcome, observation window, treatment cost accounting, and decision rule.

## Notable exploratory findings

- **Equipment age** was the leading XGBoost feature by share of total gain (8.60%), followed by **tenure (`months`)** (5.69%), **usage change (`change_mou`)** (4.55%), and **minutes of use** (3.68%). Feature importance is **not** causal evidence.
- Median device age was 366 days for churners versus 310 for retained customers; median monthly minutes of use were 330 versus 381.
- Churn by device-age quintile is **non-monotonic**: 37.4% for 189 to 300 days, jumping to 59.5% for 301 to 395 days. This is an unexplained pattern worth investigating (for example, contract or upgrade-eligibility timing), but no explanation has been tested.
- Mid-age devices (301 to 395 days) with low monthly usage had **61.84%** observed churn (7,894 customers). Thresholds are exploratory sample cut-points, not validated production rules.
- Segment analysis covers 99,642 customers; 358 are excluded for missing device age (1) or monthly usage (357).

## Repository structure

```text
telecom-churn-prediction-xgboost/
├── docs/
│   ├── METHODOLOGY.md
│   ├── PROJECT_BRIEF.md
│   └── presentation.pdf
├── data/
│   ├── input/
│   │    ├── Client.csv
│   │    └── Record.csv               
│   └── output/
│       ├── feature_importance.csv
│       ├── final_test_metrics.csv
│       ├── financial_scenarios.csv
│       ├── financial_sensitivity.csv
│       ├── model_comparison.csv
│       ├── pilot_financial_scenarios.csv
│       ├── pilot_power.csv
│       ├── segment_analysis.csv
│       ├── targeting_strategy_comparison.csv
│       └── test_targeting.csv
├── notebooks/
│   └── telecom_churn_prediction_xgboost.ipynb
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore
```

**Data note:** raw files are expected at `data/input/`. Confirm you have the right to use and redistribute them before publishing them in a public repository; the MIT license in this repository covers the code and documentation, not necessarily the data.

## Reproducing the analysis

1. Obtain copies of `Client.csv` and `Record.csv`.
2. Open `notebooks/telecom_churn_prediction_xgboost.ipynb` in Kaggle, VS Code, or Colab. Attach the files as a private dataset or edit `DATA_PATH` for your environment.
3. Enable a CUDA GPU. The reported run used Python 3.13 and XGBoost 3.4.1 on a Kaggle machine with Tesla T4 GPUs; the configuration uses a single device (`tree_method="hist"`, `device="cuda:0"`). PyTorch is imported only for the GPU availability check.
4. Run all cells in order. Every preprocessing fit must stay confined to the training split.
5. The notebook exports ten CSV tables to `final_assignment_outputs/`; copy them to `data/output/` and inspect figures and financial assumptions before reusing them.

Seeds are fixed (`random_state=42`), but GPU histogram training is not guaranteed to be bit-identical across hardware or library versions, so expect small differences in the third or fourth decimal place.

## Limitations and responsible use

- **Moderate discrimination:** test ROC-AUC of 0.6945 is not a production-readiness guarantee.
- **Sampling:** observed churn of about 50% likely does not match deployment prevalence; calibration and precision at a given budget would differ in production.
- **Validation design:** random split, not time-based. Temporal stability and leakage risk (for example, features measured close to the churn window) need checking before deployment.
- **Business assumptions:** revenue proxies are not profit; costs, margins, retention duration, and effect sizes are hypothetical.
- **No treatment data:** churn-risk prediction is not uplift modeling. High risk does not imply high responsiveness to outreach. Only a randomized experiment can establish incremental retention effect.
- **Single manual configuration:** the "tuned" model is one hand-adjusted configuration, not a systematic hyperparameter search.
- **Fairness and privacy:** demographic and proxy variables (for example ethnicity, household composition, income) are present in the inputs. Review them for fairness, legal basis, consent, and operational appropriateness before any individualized targeting.

## Suggested next steps

- Time-based backtesting on data with real-world churn prevalence, plus probability recalibration.
- Systematic hyperparameter search and an ablation excluding demographic and sensitive proxy variables.
- Uplift or causal modeling once pilot data exists.
- Run the randomized pilot described above before any scaled rollout.

## Further reading

- GSMA Intelligence (2025), *The mobile churn challenge: where loyalty is lowest and four recommendations for operators*. https://www.gsmaintelligence.com/research/the-mobile-churn-challenge-where-loyalty-is-lowest-and-four-recommendations-for-operators
- McKinsey & Company (2024), *The network is the product: How AI can put telco customer experience in focus*. https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-network-is-the-product-how-ai-can-put-telco-customer-experience-in-focus
- Ofcom (2025), *Telecoms companies up their game on complaint handling, but more to do to keep customers satisfied*. https://www.ofcom.org.uk/phones-and-broadband/service-quality/telecoms-companies-up-their-game-on-complaint-handling-but-more-to-do-to-keep-customers-satisfied

## Documentation

- [Methodology and reproducibility notes](docs/METHODOLOGY.md)
- [Project brief](docs/PROJECT_BRIEF.md)

## Citation

If you use this project in academic research, publications, educational materials, or derivative works, please cite it and credit the original author. The [`CITATION.cff`](CITATION.cff) file enables GitHub's **"Cite this repository"** button (BibTeX, APA, and other formats).

> Arain, S. U. R. (2026). *Telecom Customer Churn Prediction & Retention Analytics* (Version 1.0) [Software]. https://github.com/sanaurrehmanarain/telecom-churn-prediction-xgboost

```bibtex
@software{arain_2026_telecom_churn,
  author  = {Arain, Sana Ur Rehman},
  title   = {Telecom Customer Churn Prediction \& Retention Analytics},
  year    = {2026},
  version = {1.0},
  url     = {https://github.com/sanaurrehmanarain/telecom-churn-prediction-xgboost},
  license = {MIT}
}
```

| | |
|---|---|
| **Author** | Sana Ur Rehman Arain |
| **Role** | Data Scientist |
| **GitHub** | [@sanaurrehmanarain](https://github.com/sanaurrehmanarain) |
| **Contact** | sana.arain.work@gmail.com |

## License

Released under the **MIT License**; see [LICENSE](LICENSE). The license requires that the original copyright notice be retained in copies of the software.

---

<p align="center">
⭐ If this project was useful to you, consider starring the repo. It helps others discover it.
</p>
