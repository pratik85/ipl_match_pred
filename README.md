# IPL Win Probability Predictor

**Overview**
- **Purpose:** Predict ball-by-ball win probability for the batting team in IPL second innings using a trained logistic regression model.
- **Model file:** `pipe.pkl` (saved pipeline with OneHotEncoder + LogisticRegression).

**Repository Files**
- `matches.csv`, `deliveries.csv`: source datasets.
- `IPL_win_predictor.ipynb`: original notebook used for exploration and feature engineering.
- `app.py`: data-processing and model training script (produces `pipe.pkl`).
- `app_ui.py`: Streamlit web UI for live predictions.
- `requirements.txt`: Python dependencies.

**Requirements**
- Python 3.8+ recommended.
- Install dependencies:

```powershell
pip install -r requirements.txt
```

**How to train / regenerate model**
- Run the training script which loads CSVs, performs feature engineering, trains a Logistic Regression model, and saves the pipeline:

```powershell
python app.py
```

- After successful run, `pipe.pkl` will be created in the project folder.

**Run the Streamlit UI**
- Start the web app with Streamlit (do NOT run `app_ui.py` with plain `python`):

```powershell
streamlit run app_ui.py
```

- The UI lets you select:
  - **Batting Team**, **Bowling Team**, **City**
  - **Target Score** (first-innings total)
  - **Overs Completed** (integer 0–20) and **Balls in Over** (0–6)
  - **Current Score** and **Wickets Fallen**

- Click **Predict Win Probability** to load `pipe.pkl` and show win / lose probabilities and a bar chart.

**Notes & Tips**
- Ensure `pipe.pkl` exists before running `app_ui.py`. If missing, run `python app.py` to generate it.
- Dataset filters in preprocessing restrict teams and ignore matches with D/L applied; predictions reflect training data distribution.
- Predictions are probabilistic estimates and may not reflect live match nuances.

**Quick Commands**
- Install deps: `pip install -r requirements.txt`
- Train model: `python app.py`
- Run UI: `streamlit run app_ui.py`

**License & Contact**
- For questions or improvements, open an issue or contact the project owner.
