import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
    return pipe

# App title and description
st.set_page_config(page_title="IPL Win Predictor", layout="wide")
st.title("🏏 IPL Match Win Predictor")
st.markdown("---")

# Load model
pipe = load_model()

# Define teams and cities
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = [
    'Bangalore',
    'Hyderabad',
    'Mumbai',
    'Delhi',
    'Kolkata',
    'Chennai',
    'Jaipur',
    'Pune',
    'Chandigarh',
    'Rajkot',
    'Visakhapatnam',
    'Cuttack',
    'Dharamsala',
    'Raipur',
    'Nagpur',
    'Indore',
    'Navi Mumbai'
]

# Create columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Match Details")
    batting_team = st.selectbox("Batting Team", teams)
    bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team])
    city = st.selectbox("City", cities)

with col2:
    st.subheader("📊 Match Status")
    target_score = st.number_input("Target Score (First Innings Total)", min_value=50, max_value=250, step=1, value=160)
    col_overs, col_balls = st.columns(2)
    with col_overs:
        overs_completed = st.slider("Overs Completed", min_value=0, max_value=20, step=1, value=10)
    with col_balls:
        balls_in_over = st.slider("Balls in Over", min_value=0, max_value=6, step=1, value=0)
    wickets_fallen = st.slider("Wickets Fallen", min_value=0, max_value=10, step=1, value=3)

st.markdown("---")

# Calculate derived features
col1, col2, col3, col4 = st.columns(4)

with col1:
    balls_completed = (overs_completed * 6) + balls_in_over
    balls_left = 120 - balls_completed
    st.metric("Balls Left", balls_left)

with col2:
    wickets_left = 10 - wickets_fallen
    st.metric("Wickets Left", wickets_left)

with col3:
    current_score = st.number_input("Current Score (in 2nd Innings)", min_value=0, max_value=target_score-1, step=1, value=80)
    st.metric("Current Score", current_score)

with col4:
    runs_left = target_score - current_score
    st.metric("Runs Needed", runs_left)

st.markdown("---")

# Calculate run rates
if balls_left > 0:
    cur_run_rate = (current_score * 6) / (120 - balls_left)
    req_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Run Rate", f"{cur_run_rate:.2f}")
    with col2:
        st.metric("Required Run Rate", f"{req_run_rate:.2f}")

st.markdown("---")

# Predict button
if st.button("🎯 Predict Win Probability", use_container_width=True):
    
    if balls_left <= 0:
        st.error("⚠️ Match already completed! Balls left must be greater than 0.")
    else:
        # Create input dataframe for prediction
        input_data = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target_score],
            'cur_run_rate': [cur_run_rate],
            'req_run_rate': [req_run_rate]
        })
        
        # Get prediction
        prediction = pipe.predict(input_data)[0]
        probability = pipe.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("🏆 Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            win_prob = probability[1] * 100
            lose_prob = probability[0] * 100
            
            st.metric(f"{batting_team} Win Probability", f"{win_prob:.2f}%")
        
        with col2:
            st.metric(f"Lose Probability", f"{lose_prob:.2f}%")
        
        # Prediction bar chart
        st.markdown("---")
        pred_df = pd.DataFrame({
            'Outcome': ['Win', 'Lose'],
            'Probability': [win_prob, lose_prob]
        })
        
        st.bar_chart(pred_df.set_index('Outcome'))
        
        # Result summary
        st.markdown("---")
        if prediction == 1:
            st.success(f"✅ **{batting_team}** is likely to **WIN** the match!")
        else:
            st.error(f"❌ **{batting_team}** is likely to **LOSE** the match!")

st.markdown("---")
st.info("💡 **Note:** This prediction is based on historical IPL match data and machine learning model. Actual match outcomes may vary.")
