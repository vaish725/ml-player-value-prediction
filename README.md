# ⚽ Machine Learning for Player Market Value & Injury Risk Prediction

This project uses machine learning to analyze and predict two key aspects in professional football:
- 📈 **Player Market Value** (Regression)
- 🏥 **Injury Risk Assessment** (Classification – upcoming)

The goal is to assist football clubs and analysts with **data-driven decision-making** in scouting, valuation, and risk management.

---

## 🧠 Project Objectives

- Predict a player's current market value in euros based on attributes like age, position, and performance history.
- Predict the risk of a player getting injured in the next 30 days (to be implemented).
- Build interpretable models using structured data from real football datasets.

---

## 📊 Datasets Used

| Dataset Name        | Description                                      |
|---------------------|--------------------------------------------------|
| **Transfermarkt**   | Player profiles, market values, positions        |
| **FIFA + Simulated**| Injury logs and motion parameters for testing    |

- The datasets were cleaned and enhanced with additional features such as `age_group`, `position_encoded`, and `value_log`.
- Source: [Kaggle Transfermarkt Dataset](https://www.kaggle.com/) + internal FIFA test logs.

---
## 💻 How to Run This Project

1. Clone this repo:
   ```bash
   git clone https://github.com/vaish725/ml-player-value-prediction.git
   cd ml-player-value-prediction
   ```
2. Open the notebooks in Google Colab or Jupyter Notebook:

- Start with:

  - 01_EDA_and_FeatureEngineering_TransfermarktPlayers.ipynb

  - 01_EDA_and_FeatureEngineering_FIFA_InjuryValueAnalysis.ipynb

- Then run:

  - 02_Regression_PlayerMarketValue_Transfermarkt.ipynb

All plots will be auto-generated. Evaluation metrics and feature importance are included.

------------

## 🙋‍♀️ Author
Vaishnavi Kamdi

CSCI6364.80 — Machine Learning

Spring 2025

The George Washington University
