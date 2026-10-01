#  Hourly Energy Consumption Forecasting (PJM Grid)

##  Project Overview
This project focuses on predicting hourly energy consumption (MW) for the PJM grid. Accurate forecasting is critical for grid stability, dynamic pricing, and preventing power outages. The objective was to build a robust machine-learning model capable of capturing complex daily and yearly seasonalities from historical time-series data.

##  Tech Stack & Tools
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
* **Deployment:** Streamlit (App built for dynamic user inputs)

##  Key Methodology & Feature Engineering
Unlike traditional standard ML, time-series forecasting requires strict chronological integrity.
* **Chronological Split:** Maintained a strict 80/20 train-test split without shuffling to prevent data leakage (look-ahead bias).
* **Feature Extraction:** Decomposed raw datetime timestamps into behavioral predictors (`Month`, `Hour`, `Is_Weekend`) to help the model understand human and industrial consumption patterns.
* **One-Hot Encoding:** Transformed categorical seasonal data for machine-readability.
* **Outlier Treatment:** Handled mathematical anomalies (e.g., sensor failures) via localized linear interpolation while preserving business outliers (extreme summer/winter peak demands).

##  Model Performance & Evaluation
I established a baseline using an automated statistical model (Auto-ARIMA) before transitioning to an advanced Machine Learning engine. 

1. **Baseline Model (Auto-ARIMA):** 
   * Captured a general downward trend but failed against complex overlapping temporal patterns.
   * **Accuracy:** ~35.41% (MAPE: 64.59%)

2. **Champion Model (XGBoost Regressor):**
   * Effectively mapped temporal features and handled high volatility.
   * **Base Accuracy:** 90.47%

3. **Hyperparameter Tuning (GridSearchCV):**
   * Implemented `TimeSeriesSplit(n_splits=3)` to validate folds chronologically and prevent time-travel data leakage.
   * **Optimal Parameters:** `n_estimators=500`, `learning_rate=0.01`, `max_depth=5`
   * **Final RMSE:** 690.26 MW
   * ** Final Accuracy:** 90.57% (MAPE: 9.43%)

## 💡 Business Impact
The final XGBoost model successfully reduces the forecasting error margin to under 10%. Deployed via a Streamlit interface, this model enables grid operators to input prospective dates and times to receive highly accurate demand projections, facilitating proactive load management and optimized energy distribution.
