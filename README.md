# Algorithmic Fairness Audit of a Recidivism Prediction Model

An independent audit of a machine learning model trained on the real-world COMPAS dataset, examining whether — and how — it produces racially disparate outcomes in the criminal justice system.

This project began as a capstone during my Data Science coursework at UIUC and was independently expanded with a full fairness audit, regulatory compliance mapping, and a responsible AI governance framework.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/angi3la/algorithmic-fairness-audit-compas/blob/main/IS467_Final_Capstone_angelag5-2.ipynb)

📄 [Full report (PDF)](./full_report.pdf)

---

## Summary

Using the ProPublica COMPAS dataset (7,214 real criminal justice records from Broward County, FL), I trained a Random Forest classifier to predict two-year recidivism. While the model looked reasonable on the surface — 65% accuracy, 0.68 AUC-ROC — a deeper fairness audit revealed it produced a **41% higher false positive rate for Black defendants** than for white defendants.

Using SHAP explainability, I showed that race operated as the model's **third most influential feature independently** — not merely through proxy variables — and traced a **71% racial gap** in prior-conviction counts driving the top predictor. Counterfactual recourse testing further showed that wrongly flagged individuals had **no actionable path** to change their outcome.

I mapped these findings to three real regulatory frameworks — the **EU AI Act**, the **U.S. Algorithmic Accountability Act**, and the **Illinois Pretrial Fairness Act** — producing a full compliance gap analysis, a 12-item pre-deployment governance checklist, and a plain-language disclosure notice for affected stakeholders.

---

## Key Findings

| Metric | African-American | Caucasian |
|---|---|---|
| Recidivism rate (actual) | 51.4% | 39.4% |
| Avg. COMPAS score (1–10) | 5.37 | 3.74 |
| False positive rate | 36.2% | 25.6% |
| False negative rate | 34.4% | 50.8% |
| Predictive parity | 65.2% | 52.6% |
| Avg. prior convictions | 4.44 | 2.59 (71% gap) |

- **Model performance:** 65% accuracy, 0.68 AUC-ROC (Random Forest, 100 trees, `random_state=42`)
- **SHAP feature importance:** `priors_count` > `age` > `race_African-American` — race operates as an independent, direct driver of predictions, not just through correlated variables
- **Recourse test:** For a 22-year-old Black defendant wrongly flagged as high-risk, no single actionable change (age, priors, or race alone) reversed the prediction — only combining zero priors with an unattainable age shift did
- **Intersectional analysis:** Black men accounted for 123 of the test set's false positives vs. 59 for white men

---

## Methodology

1. **EDA** on demographic and outcome variables across the full dataset
2. **Model training** — Random Forest classifier (age, priors, juvenile record, sex, race as features), 80/20 train/test split
3. **Fairness metrics** — demographic parity, false positive/negative rate, predictive parity, disaggregated by race
4. **Explainability** — SHAP (TreeExplainer) to identify and rank feature influence, including directional effects
5. **Counterfactual recourse testing** — simulating feature changes to test whether wrongly flagged individuals have any actionable path to a different outcome
6. **Regulatory mapping** — cross-referencing findings against the EU AI Act, U.S. Algorithmic Accountability Act, and Illinois Pretrial Fairness Act

Data loads directly from the [ProPublica COMPAS GitHub repository](https://github.com/propublica/compas-analysis) — no local files needed. All random states are fixed at 42 for reproducibility.

---

## Why This Matters: An Ethical Framework

The three standard fairness metrics — demographic parity, false positive rate balance, and predictive parity — **cannot all be satisfied simultaneously** when base recidivism rates differ between groups (a mathematical result formalized by Chouldechova, 2017). That makes the central question ethical, not just technical: *which metric should be prioritized, and who bears the cost?*

A **utilitarian** framing would accept racial disparities as a tradeoff for overall accuracy. A **deontological** framing — which this audit adopts — holds that using race as a predictive feature is impermissible regardless of any accuracy benefit, since it means judging individuals by immutable characteristics they did not choose. In a domain where model errors mean wrongful detention or denied bail, the *distribution* of errors matters as much as their frequency.

---

## Regulatory Compliance Gap Analysis

| Finding | Requirement | Regulation | Status |
|---|---|---|---|
| Race is 3rd most influential feature | Bias examination/mitigation in training data | EU AI Act Art. 10(2)(f)(g) | Non-compliant |
| 41% higher FPR for Black defendants | Pre-deployment impact assessment | Algorithmic Accountability Act (2023) | Non-compliant |
| No pre-deployment bias audit | Risk management system required | EU AI Act Art. 9(8) | Non-compliant |
| No recourse path for wrongly flagged individuals | Right to explanation/contestation | GDPR Art. 22 | Non-compliant |
| `priors_count` encodes over-policing disparities | Unbiased training data required | EU AI Act Art. 10(2)(f) | Non-compliant |
| No individualized explanation for defendants | Individualized pretrial assessment | Illinois Pretrial Fairness Act | Non-compliant |

*Full 12-item compliance table, article-by-article analysis, and cross-jurisdictional comparison available in the [full report](./full_report.pdf).*

---

## Deliverables

- 12-item **pre-deployment governance checklist** for criminal justice predictive models
- **Plain-language disclosure notice** explaining, in accessible terms, how a risk score works and how to contest it
- Full **regulatory comparison** across EU, U.S. federal, and Illinois state frameworks

---

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `SHAP` · `matplotlib` · `seaborn`

---

## References

Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). *Machine Bias.* ProPublica.
Chouldechova, A. (2017). *Fair Prediction with Disparate Impact.* Big Data, 5(2), 153–163.
Kearns, M., & Roth, A. (2019). *The Ethical Algorithm.* Oxford University Press.

*Full reference list in the [full report](./full_report.pdf).*
