# Algorithmic Fairness in Binary Classification

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)

A fairness-aware machine learning pipeline that trains a binary classifier on the [UCI Adult Income](https://archive.ics.uci.edu/dataset/2/adult) dataset, audits its performance across demographic groups, and applies reweighting as a mitigation technique.

## Dataset

| File | Records | Description |
|------|---------|-------------|
| `adult.data` | 32,561 | Training set — 14 attributes + income label |
| `adult.test` | 16,281 | Test set — same schema, labels have trailing period |

The dataset was extracted from the 1994 U.S. Census and contains **45,222 clean records** after removing unknowns. The prediction task is binary: whether an individual earns more than $50K/year. The positive class represents roughly 24% of the data.

## Pipeline

1. **Data Loading & Cleaning** — Parse both splits, strip label artifacts, drop rows with missing values encoded as `?`, and create a binary target.
2. **EDA** — Distribution charts for sex, race, and age group; positive-rate analysis per group; printed underrepresentation summary table identifying which demographic subgroups are smallest and least likely to be classified as high earners.
3. **Preprocessing** — One-hot encoding of categoricals, standard scaling of continuous features, and preservation of protected attribute arrays aligned with the test set.
4. **Baseline Model** — Logistic regression trained to maximise accuracy with no fairness constraints. Full classification report printed.
5. **Fairness Audit** — Precision, recall, F1, and positive prediction rate computed separately for each value of sex, race, and age group. Grouped bar charts visualise disparities.
6. **Mitigation via Reweighting** — Sample weights are computed as $w_{g,y} = N \;/\; (|G| \cdot 2 \cdot N_{g,y})$ so that every (group, label) combination contributes equally during training. The model is retrained with these weights.
7. **Before/After Comparison** — Side-by-side bar charts for recall and positive prediction rate, an overall metrics delta table, and a demographic parity gap reduction percentage.
8. **Discussion** — Analysis of the accuracy–fairness trade-off, limitations of reweighting, and pointers to more advanced techniques.

## Libraries

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Logistic regression, scaling, evaluation metrics |
| `matplotlib` | Visualisations |
| `seaborn` | Color palettes and plot styling |

## How to Run

```bash
git clone https://github.com/<your-username>/ai-fairness-adult.git
cd ai-fairness-adult

pip install pandas numpy scikit-learn matplotlib seaborn

code ai_fairness.ipynb
```

Place `adult.data` and `adult.test` in the same directory as the notebook before running.
