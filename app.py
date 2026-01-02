import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pickle

# Load data
matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

print("Matches shape:", matches.shape)
print("Deliveries shape:", deliveries.shape)

# Calculate total runs for each inning in each match
totalrun_df = deliveries.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()

# Consider only first innings and add one for prediction
totalrun_df = totalrun_df[totalrun_df['inning'] == 1]
totalrun_df['total_runs'] = totalrun_df['total_runs'].apply(lambda x: x + 1)

# Merge with matches dataframe
match_df = matches.merge(totalrun_df[['match_id', 'total_runs']],
                         left_on='id', right_on='match_id')

# Define frequent teams
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

# Replace old team names
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

# Filter matches with only frequent teams
match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

# Remove DL method matches
match_df = match_df[match_df['dl_applied'] == 0]

# Keep only relevant columns
match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]

# Merge with deliveries
delivery_df = match_df.merge(deliveries, on='match_id')

# Filter for second innings only
delivery_df = delivery_df[delivery_df['inning'] == 2]

# Calculate current score (cumulative runs in second innings)
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()

# Calculate runs left
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']

# Calculate balls left
delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])

# Convert player_dismissed to binary
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: x if x == "0" else "1")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')

# Calculate wickets fallen
wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
delivery_df['wickets'] = 10 - wickets

# Calculate run rates
delivery_df['cur_run_rate'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
delivery_df['req_run_rate'] = (delivery_df['runs_left'] * 6) / (delivery_df['balls_left'])

# Create result column (1 if batting team won, 0 otherwise)
def resultfun(row):
    return 1 if row['batting_team'] == row['winner'] else 0

delivery_df['result'] = delivery_df.apply(resultfun, axis=1)

# Prepare final dataframe
final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left',
                        'balls_left', 'wickets', 'total_runs_x', 'cur_run_rate',
                        'req_run_rate', 'result']]

# Drop null values
final_df = final_df.dropna()

# Remove rows where balls_left is 0
final_df = final_df[final_df['balls_left'] != 0]

print(f"Final dataframe shape: {final_df.shape}")

# Prepare features and target
data = final_df.copy()
test = data['result']
train = data.drop(['result'], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=1)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Create column transformer for categorical encoding
cf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first'), 
     ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

# Create and train logistic regression pipeline
pipe = Pipeline(steps=[
    ('step1', cf),
    ('step2', LogisticRegression(solver='liblinear'))
])

pipe.fit(X_train, y_train)

# Evaluate model
y_pred = pipe.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy}")

# Save the trained model
pickle.dump(pipe, open('pipe.pkl', 'wb'))
print("Model saved as pipe.pkl")
