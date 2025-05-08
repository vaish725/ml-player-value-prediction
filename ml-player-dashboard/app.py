import streamlit as st
import joblib
import numpy as np
import time
from streamlit_extras.badges import badge
from streamlit_extras.metric_cards import style_metric_cards
import pandas as pd
import matplotlib.pyplot as plt

# Load trained models for value prediction and injury classification
value_model = joblib.load("../models/rf_market_value_model.pkl")
injury_model = joblib.load("../models/rf_injury_risk_model.pkl")

# Setup streamlit page with wide layout and icon
st.set_page_config(page_title="Football Player Insights Dashboard", layout="wide", page_icon="⚽")

st.markdown("""
<style>
    .main {background-color: #fafafa;}
    .block-container {padding-top: 2rem;}
    .stButton>button {border-radius: 0.5rem; background-color: #2e8b57; color: white;}
    .stButton>button:hover {background-color: #1e6640;}
</style>
""", unsafe_allow_html=True)

st.title("🏟️ Football Player Insights Dashboard")
with st.expander("ℹ️ About this App", expanded=False):
    st.markdown("""
    This dashboard empowers football analysts, scouts, and data enthusiasts with:
    - **Player Market Value Estimation** using physical and historical performance data
    - **Injury Risk Prediction** based on historical injuries and playing history

    All predictions are powered by **Random Forest models** trained on player data extracted from [Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) and [FIFA datasets](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset).

    Developed by Vaishnavi Kamdi.
""")

badge(type="github", name="vaish725/ml-player-value-prediction")
st.caption("A modern Streamlit app to evaluate a football player's **market worth** and **injury risk** using machine learning models.")
st.markdown("---")

# Helper function to format predicted value in Euro currency
def euro_format(amount):
    """
    Format a number as Euro currency with thousands separator and no decimals.
    """
    return f"€ {amount:,.0f}"

# ----------------------
# Section Functions
# ----------------------

def market_value_section():
    st.subheader("💶 Market Value Estimator")
    st.markdown("Estimate a player's current market value using their physical attributes and previous peak valuation.")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            player_age = st.slider("Player Age", 16, 45, 26)
            player_height = st.slider("Player Height (cm)", 160, 210, 180)
        with col2:
            player_position = st.selectbox("Playing Position", ["Goalkeeper", "Defender", "Midfielder", "Forward"])
            max_historic_value = st.number_input("Historic Highest Value (EUR)", value=750000, step=10000)

    position_encoder = {"Goalkeeper": 0, "Defender": 1, "Midfielder": 2, "Forward": 3}
    encoded_position = position_encoder[player_position]
    market_input = np.array([[player_age, player_height, encoded_position, max_historic_value]])

    if st.button("📈 Estimate Value"):
        with st.spinner("Running valuation model..."):
            time.sleep(1.2)
            value_prediction = value_model.predict(market_input)[0]
        st.metric(label="Estimated Market Value", value=euro_format(value_prediction), delta=None)
        style_metric_cards()

def injury_risk_section():
    st.subheader("🩺 Injury Risk Estimator")
    st.markdown("Predict a player's injury probability based on historical injuries and performance metrics.")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            age_in = st.slider("Age", 16, 45, 26)
            bmi_in = st.number_input("BMI", value=22.5, step=0.1)
            avg_injury_days = st.slider("Avg. Days Injured (Past Seasons)", 0, 180, 25)
        with col2:
            avg_games_played = st.slider("Avg. Games/Season", 0, 45, 20)
            total_injury_days = st.slider("Cumulative Injury Days", 0, 1000, 200)
            last_season_days = st.slider("Days Injured Last Season", 0, 200, 40)

    risk_input = np.array([[age_in, bmi_in, avg_injury_days, avg_games_played, total_injury_days, last_season_days]])

    if st.button("🧠 Predict Injury Risk"):
        with st.spinner("Evaluating injury risk..."):
            time.sleep(1.2)
            risk_probability = injury_model.predict_proba(risk_input)[0][1]
        st.metric(label="Injury Risk Probability", value=f"{risk_probability:.2%}", delta=None)
        st.progress(int(risk_probability * 100))

        if risk_probability >= 0.7:
            st.error("❗ High Injury Risk — exercise caution with this player.")
        elif risk_probability >= 0.4:
            st.warning("⚠️ Moderate Risk — player may require close monitoring.")
        else:
            st.success("✅ Low Injury Risk — player appears physically resilient.")

def correlation_section():
    """
    Section to analyze and visualize the correlation between market value and injury risk predictions.
    """
    st.subheader("📊 Correlation Between Model Outcomes")
    st.markdown("""
    This section explores the relationship between the predicted **market value** and the **injury risk probability** for a sample of players.
    - The injury risk model outputs a probability (between 0 and 1) for each player.
    - The scatter plot below shows each player's predicted market value (x-axis) vs. their predicted injury risk probability (y-axis).
    - The correlation coefficient quantifies the linear relationship between these two outcomes.
    """)

    # Load a sample of players from the Transfermarkt dataset
    df_tm = None
    try:
        df_tm = pd.read_csv("../datasets/cleaned_tm_players_dataset_v3_with_features.csv")
    except Exception as e:
        st.error(f"Could not load player dataset: {e}")
        return

    # For demonstration, sample 200 players
    df_sample = df_tm.sample(n=min(200, len(df_tm)), random_state=42)

    # Prepare features for both models
    # Market value model features
    mv_features = df_sample[["age", "height_in_cm", "position_encoded", "highest_market_value_in_eur"]].values
    # Injury risk model features (use default/fallback if not present)
    ir_feature_names = ["age", "bmi", "avg_days_injured_prev_seasons", "avg_games_per_season_prev_seasons", "total_days_injured", "season_days_injured_prev_season"]
    for col in ir_feature_names:
        if col not in df_sample:
            df_sample[col] = 0  # fallback if missing
    ir_features = df_sample[ir_feature_names].values

    # Get predictions
    mv_preds = value_model.predict(mv_features)
    ir_probs = injury_model.predict_proba(ir_features)[:, 1]

    # Calculate correlation
    corr = np.corrcoef(mv_preds, ir_probs)[0, 1]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mv_preds, ir_probs, alpha=0.6, color="#2e8b57")
    ax.set_xlabel("Predicted Market Value (EUR)")
    ax.set_ylabel("Predicted Injury Risk Probability")
    ax.set_title("Market Value vs. Injury Risk Probability")
    st.pyplot(fig)

    st.markdown(f"**Correlation coefficient:** {corr:.3f}")
    st.caption("A value close to 0 means little/no linear relationship; closer to 1 or -1 means strong positive/negative correlation.")

# Sidebar navigation for modules
section = st.sidebar.radio("📋 Choose Module:", [
    "Estimate Market Value", "Assess Injury Risk", "Outcome Correlation"
])

if section == "Estimate Market Value":
    market_value_section()
elif section == "Assess Injury Risk":
    injury_risk_section()
elif section == "Outcome Correlation":
    correlation_section()

st.caption("⚙️ Powered by machine learning trained on > 1,000 player samples.")
#footer line
st.markdown("---")
st.caption("📌 *Disclaimer: This is an academic prototype. Predictions are based on historical data and are not to be used as the sole basis for professional decisions.*")

#------------------------------------------------------
# FOOTER NOTE:
# This dashboard is inspired by Streamlit example apps from https://docs.streamlit.io/ and https://streamlit.io/gallery
# Styling and layout enhancements customized for this project by Vaishnavi Kamdi.
