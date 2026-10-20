# Project Brief: Telecom Customer Churn Prediction & Retention Analytics

**Author:** Sana Ur Rehman Arain · **Version:** 1.0 · **Date:** 20 October 2026
**Status:** Retrospective proof of concept. No retention intervention has been run.

## 1. The task

A telecom provider holds historical customer profiles and service-usage records but lacks an analytical approach for identifying customers likely to churn and for planning targeted retention outreach. The task is to build a reproducible proof of concept that turns this data into predictive insight and a measurable business recommendation.

## 2. Objectives

- Inspect and merge customer and usage data reliably.
- Explore patterns associated with churn, including equipment age, tenure, usage change, and revenue.
- Train and compare baseline and non-linear classifiers with leakage-safe validation.
- Evaluate risk ranking and value-based prioritization at realistic outreach budgets.
- Quantify costs, contribution, and break-even intervention effectiveness under transparent, hypothetical assumptions.
- Propose a randomized pilot to measure whether outreach actually reduces churn.

## 3. Data and target

| Item | Detail |
|---|---|
| `Client.csv` | 100,000 records × 50 columns (profile, demographics, handset, billing history) |
| `Record.csv` | 100,000 records × 51 columns (usage, service quality, churn label) |
| Merge | One-to-one on unique `Customer_ID`, giving 100,000 × 100 |
| Target `churn` | Churn between 31 and 60 days after an observation date (source documentation); 49.56% positive |
| Predictors | 98 original features (77 numeric, 21 categorical), expanded to 272 after preprocessing |
| Splits | Stratified random: 70,000 train / 15,000 validation / 15,000 test |

## 4. Approach

1. Validate and merge the tables; profile missingness (342,969 missing cells; six columns above 30% missing).
2. Explore churn by device age, usage, revenue, and combined segments.
3. Preprocess with train-only fitting: median imputation with missing indicators, constant imputation and one-hot encoding for categoricals.
4. Fit a Logistic Regression baseline and GPU XGBoost; select the final XGBoost on validation ROC-AUC with early stopping.
5. Report final performance once on the held-out test set.
6. Evaluate risk-only and value-aware targeting, then build illustrative campaign economics and a pilot design.

## 5. Predictive results

| Model | Validation ROC-AUC | Validation PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.6328 | 0.6081 |
| Original XGBoost | 0.7013 | 0.6914 |
| Tuned XGBoost | 0.7032 | 0.6923 |

The selected tuned XGBoost achieved on the **held-out test set**: ROC-AUC **0.6945**, PR-AUC **0.6816**, accuracy **64.01%**, precision 63.40%, recall 64.76%, F1 0.6407, Brier score 0.2206. The non-linear model clearly outperforms the linear baseline (about +0.07 validation ROC-AUC). Discrimination is moderate in absolute terms.

**Leading drivers (XGBoost total gain):** equipment age (8.60%), tenure (5.69%), change in minutes of use (4.55%), and minutes of use (3.68%). These are associations used by the model, not causes.

## 6. Business insights

**Targeting at a fixed budget (test set).** Contacting the top 20% by predicted risk (3,000 of 15,000 customers) captured 29.47% of observed churners (2,191 of 7,434), a lift of 1.474× over random selection (about 1,487 churners expected). Smaller budgets concentrate risk more tightly: the top 5% had 85.1% observed churn (1.716× lift). Because the sample is about 50% churn, lift is capped near 2×.

**Coverage versus value.** Re-ranking by predicted churn probability × monthly revenue proxy identified **63.3% more** retrospective churner revenue (199,533 vs 122,199 units) in the same-sized group, but reached fewer churners (1,798 vs 2,191). This shows a trade-off between churn coverage and revenue exposure; it does not demonstrate higher recoverable profit.

**Segments.** Customers with devices 301 to 395 days old and low usage had 61.84% observed churn (7,894 customers), while customers with devices up to 300 days old churned at 38% to 41%. Churn is not monotonic in device age, which merits investigation. Thresholds are exploratory.

**Campaign economics (illustrative).** Assuming 6 retained months, 60% margin, and costs of 2 (contact) + 15 (incentive) per contacted customer, the top-20% campaign costs 51,000 units and breaks even at about **11.59% relative churn reduction** (11.36% at the top 10%, 12.12% at the top 30%). At a 10% effect the campaign loses about 7,008 units; at 15% it gains about 14,987. Break-even ranges from 4.77% to 18.41% as incentive cost moves from 5 to 25 units. Real prevalence is probably lower than in this balanced sample, which would push the break-even higher.

## 7. Proposed intervention

Run a controlled experiment on a predefined eligible group:

- **Design:** random assignment to treatment (risk-ranked outreach) and control (no outreach), decided before contact. Illustrative size: 3,000 customers, 1,500 per arm.
- **Power:** at 80% power and α = 0.05, 1,500 per arm can detect an absolute churn reduction of 2.85 pp at 10% baseline churn, 3.93 pp at 20%, 4.58 pp at 30%, and 5.11 pp at 50% (10.2% to 28.6% relative). If baseline churn is low, a larger pilot is needed to detect a break-even-sized effect.
- **Pre-declared:** primary outcome (churn in a window matching the label definition), observation period, treatment cost accounting, analysis plan, and decision rule.
- **Monitor:** treatment cost, churn reduction, incremental contribution, complaints, opt-outs, and effects across customer groups.
- **Scale only if** evidence supports both operational suitability and commercial viability (incremental contribution per contacted customer is credibly positive, not merely a statistically significant churn drop).

## 8. Interpretation caveats

- The sample appears balanced for churn, so prevalence, precision, and calibration will differ in production.
- Currency, margins, and costs are unknown; all financial figures are hypothetical units.
- No treatment responses were supplied. Churn prediction is not uplift modeling; high-risk customers are not necessarily the most persuadable.
- Validation used a random split, not a chronological backtest, and no confidence intervals were computed.
- The "tuned" model is one manually adjusted configuration, not a systematic search.
- Inputs include demographic and household attributes that require fairness, privacy, and legal review before individualized targeting.
- The business scenarios are hypothetical and must not be presented as realized or guaranteed savings.

## 9. Recommended next steps

1. Obtain real campaign costs, margins, and the true churn prevalence of the eligible population.
2. Re-evaluate the model with a time-based split and recalibrate to production prevalence.
3. Audit sensitive and proxy variables; test a model without them.
4. Finalize the pilot protocol and size it from observed baseline churn.
5. After the pilot, consider uplift modeling to target customers who respond to outreach.

## 10. Deliverables

Notebook (`notebooks/telecom_churn_prediction_xgboost.ipynb`), ten result tables (`data/output/`), [README](../README.md), [methodology notes](METHODOLOGY.md), and a presentation (`docs/presentation.pdf`).
