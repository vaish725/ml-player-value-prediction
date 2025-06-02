# Sports Analytics: Predicting Football Player Market Value & Injury Risk Probability

**A Graduate Project for CSCI6364 Machine Learning**
**The George Washington University, Spring 2025**
**Author: Vaishnavi Kamdi**
**GitHub Repository:** [https://github.com/vaish725/ml-player-value-prediction.git](https://github.com/vaish725/ml-player-value-prediction.git)

---

## 🌟 Project Overview

This project leverages machine learning techniques to address two critical challenges in the professional football (soccer) industry:
1.  📈 **Player Market Value Prediction (Regression):** Estimating a player's current market value in Euros.
2.  🏥 **Player Injury Risk Assessment (Classification):** Predicting the probability of a player having a significant injury based on their historical data.

The goal is to provide data-driven insights that can assist football clubs, analysts, and scouts in making more informed decisions regarding player valuation, acquisition, and health management. This project emphasizes a full machine learning pipeline, from data exploration and feature engineering to model development, evaluation, and deployment via an interactive dashboard.

---

## 🎯 Key Objectives & Tasks

* **Market Value Prediction:**
    * Develop a regression model to accurately predict a player's current market value.
    * Target: Achieve a high R² score (e.g., >0.80) and low Mean Absolute Error (MAE).
    * Models Explored: Linear Regression, Random Forest, XGBoost.
    * Final Model Deployed: Random Forest Regressor.
* **Injury Risk Assessment:**
    * Develop a classification model to predict the probability of a player having experienced a significant injury in the previous season, based on their historical attributes and injury record.
    * Target: Achieve high predictive accuracy (e.g., AUC-ROC >0.80).
    * Models Explored: Logistic Regression, Random Forest, XGBoost.
    * Final Model Deployed: Random Forest Classifier.
* **Deliverable:** An interactive Streamlit dashboard for users to get live predictions for both market value and injury risk probability.

---

## 📊 Datasets Used

1.  **Transfermarkt Player Dataset (for Market Value Prediction):**
    * **Source:** Kaggle (derived from Transfermarkt.com).
    * **Description:** Contains player profiles, including attributes like age, height, position, and historical market values (current and peak).
    * **Key Features Used in Final Model:** `age`, `height_in_cm`, `position_encoded`, `highest_market_value_in_eur`.
2.  **FIFA Player Dataset (for Injury Risk Assessment):**
    * **Source:** Kaggle (derived from FIFA game data, enhanced with injury metrics).
    * **Description:** Includes player attributes (age, BMI), workload indicators (games played), and detailed historical injury data (days injured, cumulative injuries).
    * **Target Variable:** `significant_injury_prev_season` (binary: 1 if player had a significant injury in the previous season, 0 otherwise).
    * **Key Features Used in Final Model:** `age`, `bmi`, `avg_days_injured_prev_seasons`, `avg_games_per_season_prev_seasons`, `cumulative_days_injured`, `season_days_injured_prev_season`.

---

## ⚙️ Methodology

### 1. Data Preprocessing & Exploratory Data Analysis (EDA)
* Thorough EDA was performed on both datasets to understand distributions, identify correlations, and visualize trends.
* Data cleaning involved handling missing values (imputation/removal) and correcting data types.
* Categorical features (e.g., player position) were encoded numerically.
* Feature scaling (StandardScaler) was applied where necessary for specific models.

### 2. Feature Engineering
New features were created to capture more domain-specific insights:
* For Transfermarkt: `age_group` (categorizing players by age).
* For FIFA: `BMI` (Body Mass Index), `injury_rate` (days injured per game), `minutes_per_game`, `experience_years`.
* *Note: An initial feature `value_log` for regression was found to cause data leakage and was subsequently removed from the input features for the final market value models to ensure realistic performance.*

### 3. Model Development & Evaluation

#### a) Player Market Value Prediction (Regression)
* **Models:** Linear Regression (baseline), Random Forest Regressor, XGBoost Regressor.
* **Data Leakage:** Addressed by removing features directly derived from the target variable (like `value_log`) before final training.
* **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score.
* **Best Performing Model (Deployed):** Random Forest Regressor (achieved R² ≈ 0.82 after correcting for data leakage).

#### b) Player Injury Risk Assessment (Classification)
* **Models:** Logistic Regression (baseline), Random Forest Classifier, XGBoost Classifier.
* **Output:** The models predict the probability of a player having a significant injury in the previous season.
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC Score.
* **Best Performing Models:** Random Forest Classifier and XGBoost Classifier (achieved AUC-ROC ≈ 1.00 on the test set). Random Forest was chosen for deployment in the Streamlit app.

---

## 📈 Key Results

* **Market Value Prediction:** The Random Forest Regressor demonstrated strong performance with an **R² score of approximately 0.82** and an MAE of ~€621K, after addressing initial data leakage issues. Key predictors were `highest_market_value_in_eur` and `age`.
* **Injury Risk Prediction:**
   - **Problem reframed:** Predict next‑season injury (injury_next_season), eliminating look‑ahead bias.
   - **Deployed model:** XGBoost Classifier (leak‑free pipeline saved as models/injury_xgb_leakfree.joblib).
   - **Evaluation protocol:** 5‑fold TimeSeriesSplit (gap = 1 season) grouped by player.
   - **Cross‑validated performance:** AUC = 0.79 ± 0.06 — realistic and generalisable.
   - **Key anti‑leak measures:**
      - Dropped same‑season injury columns: season_days_injured, total_days_injured, season_days_injured_prev_season, …
      - Shifted target forward one season per p_id2.
      - Encapsulated preprocessing + model in a scikit‑learn Pipeline to prevent train‑test contamination.
  - **Why the score "dropped" from 1.00 to 0.79?**  The earlier model was unknowingly reading its own answers (data leakage). After lagging the target and using temporal CV, we obtain a trustworthy metric that should hold on unseen seasons.
---

## 🚀 Interactive Dashboard (Streamlit Application)

An interactive web application was developed using Streamlit (`app.py`) to provide a user-friendly interface for the prediction models.

* **Features:**
    * Allows users to input player attributes.
    * Provides real-time predictions for:
        * Estimated Market Value (€)
        * Injury Risk Probability (%)
    * Includes an analysis of the correlation between predicted value and injury risk.
* The deployed models (`rf_market_value_model.pkl` and `rf_injury_risk_model.pkl`) are located in the `models/` directory.

---

## 💻 How to Run This Project

### 1. Prerequisites
* Python 3.7+
* Git

### 2. Clone the Repository
```bash
git clone https://github.com/vaish725/ml-player-value-prediction.git
cd ml-player-value-prediction
```

### 3. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies
A `requirements.txt` file is provided with the necessary libraries.
```bash
pip install -r requirements.txt
```
Key libraries include: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `jupyter`, `streamlit`, `joblib`, `missingno`, `streamlit-extras`.

### 5. Run Jupyter Notebooks
The notebooks are designed to be run in sequence to understand the data processing, EDA, and modeling steps:

* **Exploratory Data Analysis & Feature Engineering:**
    1.  `01_EDA_and_FeatureEngineering_TransfermarktPlayers.ipynb`
    2.  `01_EDA_and_FeatureEngineering_FIFA_InjuryValueAnalysis.ipynb`
* **Market Value Regression Modeling:**
    3.  `02_Regression_PlayerMarketValue_Transfermarkt.ipynb` (Initial models, demonstrates data leakage issue)
    4.  `03_Retraining_Regression_NoDataLeakage_Transfermarkt.ipynb` (Final models after fixing leakage)
    5.  `04_Regression_LogTarget_Transfermarkt.ipynb` (Experiment with log-transformed target)
* **Injury Risk Classification Modeling:**
    6.  `05_Classification_InjuryRisk_FIFA.ipynb`

Open these notebooks using Jupyter Notebook or JupyterLab.

### 6. Run the Streamlit Dashboard
Ensure your trained models (`rf_market_value_model.pkl` and `rf_injury_risk_model.pkl`) are in a `models/` subdirectory relative to `app.py`. If `app.py` is in the root, the models should be in `models/`. The `app.py` provided loads them from `../models/`, implying `app.py` might be intended to be run from a subdirectory or the models are one level up. Adjust paths in `app.py` or file structure if needed.

Assuming `app.py` is in the root directory and models are in `models/` (you might need to adjust `app.py`'s model loading path from `../models/` to `models/`):
```bash
streamlit run app.py
```
The application will open in your web browser.

---
## 📑 Project Report
You can find the detailed project report describing the methodology, results, and analysis at:
```bash
report/Report Paper on Sports Analytics Predicting Player Value and Injury Risk.pdf
```
---
## 🧑‍💻 Author

* **Vaishnavi Kamdi**
    * Graduate Student, CSCI6364 Machine Learning
    * The George Washington University, Spring 2025

---
