# Paramount+ Subscriber Regression Model
### Predicting Total Signups · Feb–Apr 2026
**By Lorraine A. Wheat · May 2026**

> ⚠️ All data used in this project is **synthetic**. It was generated in Claude AI to simulate realistic Paramount+ subscriber and campaign behavior. It does not represent real Paramount+ users, revenue, or marketing performance.

---

## Project Summary

This project builds a **multivariate linear regression model** to predict campaign-level total signups using subscriber and ad spend data. The central finding is that **channel efficiency — not spend volume — is the dominant driver of signup growth.** Average ROAS outperformed every other feature by an order of magnitude, with a coefficient of 287.34 compared to near-zero effects for spend, age, plan price, and bundling status.

---

## Business Question

> *Which subscriber and campaign attributes best predict total signups — and does spending more money matter more than spending it efficiently?*

This question is directly relevant to media mix modeling and budget optimization decisions at a streaming company. The model provides a data-driven answer: **efficiency wins.**

---

## Dataset

**File:** `data/subscribers_joined.csv`
**Rows:** 3,000 · **Columns:** 21

| # | Column | Type | Notes |
|---|--------|------|-------|
| 0 | subscriber_id | object | Unique ID |
| 1 | signup_date | object | |
| 2 | plan | object | Essential, Annual, Paramount+ with Showtime |
| 3 | country | object | |
| 4 | acquisition_channel | object | |
| 5 | campaign_id | object | |
| 6 | campaign_start_date | object | |
| 7 | campaign_end_date | object | |
| 8 | primary_device | object | |
| 9 | age | int64 | |
| 10 | signup_show | object | |
| 11 | show_genre | object | |
| 12 | lifecycle_stage | object | Active, At-Risk, Churned, New |
| 13 | churn_date | object | 355 non-null (churned subs only) |
| 14 | monthly_spend_usd | float64 | |
| 15 | is_bundled | bool | |
| 16 | has_showtime_add_on | bool | |
| 17 | total_spend | float64 | Campaign-level ad spend |
| 18 | total_signups | int64 | **Target variable (y)** |
| 19 | avg_roas | float64 | Return on ad spend |
| 20 | plan_price | float64 | |

---

## Model Setup

**Target (y):** `total_signups`
**Features (X):** `total_spend`, `plan_price`, `avg_roas`, `age`, `is_bundled`

```python
features = [
    'total_spend',
    'plan_price',
    'avg_roas',
    'age',
    'is_bundled'
]

X = subs_ads[features]
y = subs_ads['total_signups']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
```

**Train / Test Split:** 80% / 20% · `random_state=42`

**Target range:** 14 – 75 · **Mean:** 33.53

---

## Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R-squared** | **0.958** | 95.8% of signup variance explained by the model |
| **RMSE** | **2.98 signups** | Average prediction error of ±3 signups |

This is a high-performing model on synthetic data. In a production setting, performance would be validated against holdout periods and additional features such as adstock decay and seasonality.

---

## Feature Importance (by Coefficient)

| Rank | Feature | Coefficient | Signal |
|------|---------|-------------|--------|
| 1 | **avg_roas** | 287.34 | ★★★★★ Dominant predictor — higher ROAS → significantly more signups |
| 2 | is_bundled | +0.020 | ★★☆☆☆ Bundled subscribers slightly associated with more signups |
| 3 | age | +0.002 | ★☆☆☆☆ Minimal positive effect |
| 4 | total_spend | +0.0009 | ★☆☆☆☆ More spend → marginally more signups; efficiency matters more than volume |
| 5 | plan_price | −0.0005 | ★☆☆☆☆ Higher plan price has a slight negative effect on signups |

**Key takeaway:** A one-unit increase in `avg_roas` predicts 287 additional signups, holding all else constant. No other feature comes close. This tells the business to optimize for channel efficiency before increasing raw spend.

---

## Correlation Analysis

| Feature | r value | Strength | Direction | Interpretation |
|---------|---------|----------|-----------|----------------|
| total_spend | 0.837 | Strong | ↑ Positive | Spend volume is the highest correlator with signup volume |
| avg_roas | 0.223 | Moderate | ↑ Positive | Better efficiency correlates with more signups |
| plan_price | 0.010 | Very Weak | ↑ Positive | Plan pricing has almost no effect on signups |
| age | 0.006 | Negligible | ↑ Positive | Subscriber age does not predict signups |
| is_bundled | −0.054 | Very Weak | ↓ Negative | Bundled subscribers slightly associated with fewer signups |

> Note: `total_spend` has a strong correlation (r = 0.837) but a near-zero regression coefficient (+0.0009). This is because correlation measures the raw relationship between two variables in isolation, while the regression coefficient controls for all other features simultaneously. When `avg_roas` is in the model, it absorbs most of the explanatory power. **Spend volume appears to matter primarily because high-spend campaigns also tend to have high ROAS — not because spend itself drives signups.**

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats
```

---

## Process & Data Limitations

**Process:** Synthetic data was generated in Claude AI, then cleaned and preprocessed in Python using pandas. Applied an 80/20 train/test split and fit a multivariate linear regression model using scikit-learn.

**Limitations:** A production model would incorporate:
- **Adstock decay** to account for carry-over advertising effects
- **Seasonality controls** for release windows and holidays
- **18–24 months of historical spend data** for more robust estimation
- **Statistical significance testing** between campaign groups to validate regression findings



---

*Lorraine A. Wheat · May 2026 · [lawwrites](https://github.com/lawwrites)*