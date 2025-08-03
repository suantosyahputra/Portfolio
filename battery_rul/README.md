# Battery Remaining Useful Life (RUL) Prediction

**Description:**  
This project predicts the remaining useful life of a battery using cycle-level discharge/charge metrics. Accurate RUL estimation helps plan maintenance, avoid unexpected failures, and optimize battery usage.

**Goal:**  
Build and evaluate regression models that forecast future battery degradation, with attention to evaluation validity (time-aware forecasting vs naive random splitting) and feature signal isolation.

---

## 📊 Dataset
- **Source:** The dataset was obtained from kaggle of Ignacio Vinuales: https://www.kaggle.com/datasets/ignaciovinuales/battery-remaining-useful-life-rul?resource=download
- To check how the dataset was built, one can go to  https://github.com/ignavinuales/Battery_RUL_Prediction 
- **Key features:**
  - Discharge Time (s)
  - Decrement 3.6–3.4V (s)
  - Max. Voltage Dischar. (V)
  - Min. Voltage Charg. (V)
  - Time at 4.15V (s)
  - Time constant current (s)
  - Charging time (s)
- **Target:** Remaining Useful Life (RUL) — already precomputed
- **Observations:** 15,064 rows, all numeric, no missing values.

---

## 🛠 Tools & Libraries
- Python, Pandas, NumPy  
- scikit-learn (Random Forest, IsotonicRegression, Ridge)  
- Matplotlib for visualization  
- StandardScaler for feature scaling

---

## 📈 Methods
1. **Data inspection** and leakage analysis (excluded `Cycle_Index` from features).  
2. **Evaluation splits:**
   - Random split (standard `train_test_split`) — interpolation-style baseline.  
   - Time-aware split (train on earlier cycles, test on later cycles) — realistic forecasting.  
3. **Baseline model:** Random Forest regressor with all features.  
4. **Ablation:** Remove `"Discharge Time (s)"` to quantify its importance.  
5. **Monotonic baseline:** Isotonic regression on `Discharge Time (s)` alone to capture rank-order relationship.  
6. **Stacked residual modeling:** Isotonic prediction + separate model (with and without rolling features) on residuals to attempt extracting secondary signal.

---

## 🚀 Results

| Method / Split & Features                                         | RMSE      | Notes |
|------------------------------------------------------------------|-----------|-------|
| **Random split, all features**                                   | ~27.87    | Optimistic; leakage/interpolation across adjacent cycles. |
| **Random split, without “Discharge Time (s)” (ablation)**        | ~27.78    | Almost unchanged—shows random split masks feature dominance. |
| **Time-aware split, all features**                               | 159.86    | Realistic forecasting baseline. |
| **Time-aware split, without “Discharge Time (s)” (ablation)**    | 159.23   | Similar to full; raw Discharge Time doesn’t add forward-generalizable signal beyond others. |
| **Isotonic regression on “Discharge Time (s)” alone (time-aware)** | 219.48   | Monotonic baseline; captures rank order but lacks nuance. |
| **Stacked isotonic + residual (original, no rolling)**           | 163.52   | Attempts decomposition; no improvement over baseline. |
| **Stacked isotonic + residual with rolling + Ridge**             | 234.85   | Added temporal context but residual noisy/harmed performance. |
| **Stacked isotonic + residual with rolling + shallow tree**      | 194.61   | Better than Ridge stacking, still worse than direct baseline. |

---

## 🔍 Feature & Signal Insights
- `Discharge Time (s)` has a **strong monotonic relationship** with RUL (Spearman ρ ≈ 0.96) but a near-zero linear correlation (Pearson r ≈ 0.012), indicating non-linear dependency.  
- Ablation shows that in a **time-aware** forecasting setting, removing `Discharge Time` barely changes performance, suggesting that its raw form doesn’t generalize much better forward than other features.  
- The random split hides this nuance, giving an overly optimistic view due to interpolation of nearby cycles.  
- The stacked isotonic + residual pipeline confirmed that isolating the monotonic component and modeling residuals independently is tricky: residuals are noisy and didn’t yield a net improvement.

---

## 🧪 Evaluation Strategy
We contrasted **random split** (easy interpolation, optimistic) with a **time-aware split** (train on earlier cycle indices, test on later ones) to reflect real-world prediction, where future cycles are unseen. The time-aware split is considered the ground-truth evaluation for deployment.

---

## 💡 Future Improvements
- **Flexible encoding of `Discharge Time (s)`** (e.g., spline basis or log/quantile transforms) within a single model to better capture its nonlinear shape jointly with other features.  
- **Temporal feature augmentation:** rolling means/std of key metrics to give the model short-term context.  
- **Gradient boosting models** with monotonic constraints (if supported) to respect known order relationships while learning interactions.  
- **Calibration & uncertainty:** explore approximate Gaussian Process approaches (e.g., kernel approximations or sparse variational GPs) to quantify predictive uncertainty.  
- **Residual refinement:** revisit residual modeling with stabilized residual signals (e.g., smoothing or grouping) if decomposed approaches are needed.

---

## 📷 Visualizations
- Predicted vs Actual RUL (time-aware split)  
- Discharge Time vs RUL (monotonic but nonlinear relationship)  
- Residual distributions  

*(Include screenshots of these plots here.)*

---

## 🔗 Links
- [Full notebook on GitHub](#)  
- [Portfolio Page](#)  

