# Heart Disease Prediction

Binary classification on the UCI Cleveland Heart Disease dataset: does feature
selection by correlation strength improve model accuracy? Four classifiers are
trained twice — once on all 13 features, once on the 9 features most correlated
with the target — and compared.

**Short answer: not uniformly.** Correlation-based feature selection helped the
three weaker models substantially and slightly hurt the strongest one.

## Results

Accuracy on a held-out 20% test split (58 samples, `random_state=35`):

| Model | All 13 features | Top 9 by correlation | Change |
|---|---|---|---|
| Logistic Regression | **0.897** | 0.879 | −0.017 |
| Decision Tree | 0.690 | 0.828 | **+0.138** |
| Random Forest | 0.828 | 0.862 | +0.034 |
| AdaBoost | 0.828 | 0.879 | +0.052 |

Best overall: logistic regression on the full feature set, 89.7% accuracy
(precision 1.00 / recall 0.78 for class 0; 0.84 / 1.00 for class 1).

### Reading the result

- **Decision Tree gained the most (+13.8pp).** A single unpruned tree overfits
  noisy low-signal columns; dropping them is effectively regularisation.
- **Logistic Regression lost slightly.** It already down-weights weak features
  via its coefficients, so removing them discards a little usable signal and
  gains nothing.
- The ensembles land in between — bagging and boosting already damp weak
  features, so the cleanup helps less than it does a lone tree.

Caveat worth stating plainly: the test set is 58 samples, so a single
reclassified patient moves accuracy by 1.7pp. Differences under ~5pp here are
not meaningfully distinguishable from noise. Treat the Decision Tree jump as
the only clearly real effect.

## Dataset

[UCI Heart Disease (Kaggle mirror)](https://www.kaggle.com/ronitf/heart-disease-uci)
— 303 samples, 13 features plus binary `target`. Included as `heart.csv`.

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type (4 values) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0, 1, 2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Major vessels (0–3) coloured by fluoroscopy |
| thal | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) |

The 9 features retained for the second pass (|corr with target| ≥ 0.20):
`age, sex, cp, thalach, exang, oldpeak, slope, ca, thal`.

## Method

1. **Integrity checks** — no missing values; one duplicate row (index 163/164)
   kept deliberately, since identical measurements across two patients are
   plausible rather than a data error.
2. **Outlier review** — boxplots by target, then z-scores on the continuous
   columns only (binary columns make z-scores meaningless). Outliers were
   inspected, not dropped.
3. **Correlation analysis** — full heatmap, then an absolute-correlation filter
   at 0.20 to pick the reduced feature set.
4. **Train/test** — 80/20 split, `random_state=35` for reproducibility.
5. **Models** — Logistic Regression, Decision Tree, Random Forest (600 trees),
   AdaBoost (25 estimators), each fit on both feature sets.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook "Heart Disease Problem.ipynb"
```

## Known issues

Kept visible rather than quietly patched, since the committed outputs were
produced by the code as it stands:

- **Unscaled features trip the solver.** `LogisticRegression` emits
  `ConvergenceWarning: lbfgs failed to converge` at the default
  `max_iter=100`, because `chol` (~126–564) and `oldpeak` (~0–6.2) differ by two
  orders of magnitude. The fix is a `StandardScaler` in a pipeline; results
  below are from the unscaled run.
- **AdaBoost rerun reuses the wrong object.** In the reduced-feature section,
  `hadab = adab.fit(...)` refits the *original* estimator and predicts with
  `adab` instead of a fresh `hadab`. The reported number is still a genuine fit
  on the 9-feature data, but the variable naming is misleading.
- No cross-validation — single split only, which is why the caveat above
  matters.

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · SciPy

## Acknowledgements

Dataset collected by Andras Janosi (Hungarian Institute of Cardiology,
Budapest), William Steinbrunn (University Hospital, Zurich), Matthias Pfisterer
(University Hospital, Basel), and Robert Detrano (V.A. Medical Center, Long
Beach & Cleveland Clinic Foundation).

Outlier-detection approach adapted from
[Ways to Detect and Remove the Outliers](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba).

## License

[MIT](LICENSE) — Jonas, 2020.
