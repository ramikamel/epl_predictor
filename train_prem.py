import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import joblib

# Load data
matches = pd.read_csv('premier-league-matches.csv')

# Create day of week column
matches['day_of_week'] = pd.to_datetime(matches['Date']).dt.day_name()

# Map full-time result to numerical values
def map_ftr(result):
    if result == 'H':
        return 1
    elif result == 'A':
        return 0
    elif result == 'D':
        return 2

matches['result'] = matches['FTR'].apply(map_ftr)

# Convert Date to datetime
matches['Date'] = pd.to_datetime(matches['Date'])

# Encode categorical variables
team_code_dict = {
    'Norwich City': 0,
    'Bradford City': 1,
    'Middlesbrough': 2,
    'Hull City': 3,
    'Birmingham City': 4,
    'Chelsea': 5,
    'Blackpool': 6,
    'Swindon Town': 7,
    'Blackburn': 8,
    'Sheffield Utd': 9,
    'Manchester Utd': 10,
    'Wolves': 11,
    'QPR': 12,
    'Brighton': 13,
    'Ipswich Town': 14,
    'Tottenham': 15,
    "Nott'ham Forest": 16,
    'Charlton Ath': 17,
    'Oldham Athletic': 18,
    'Southampton': 19,
    'Huddersfield': 20,
    'Watford': 21,
    'Fulham': 22,
    'Barnsley': 23,
    'Newcastle Utd': 24,
    'Aston Villa': 25,
    'Reading': 26,
    'Liverpool': 27,
    'Burnley': 28,
    'Wigan Athletic': 29,
    'Manchester City': 30,
    'Leicester City': 31,
    'Bournemouth': 32,
    'Everton': 33,
    'Coventry City': 34,
    'West Ham': 35,
    'Sheffield Weds': 36,
    'Derby County': 37,
    'Cardiff City': 38,
    'Leeds United': 39,
    'Crystal Palace': 40,
    'Swansea City': 41,
    'Portsmouth': 42,
    'West Brom': 43,
    'Brentford': 44,
    'Wimbledon': 45,
    'Sunderland': 46,
    'Bolton': 47,
    'Arsenal': 48,
    'Stoke City': 49
}

day_of_week_code_dict = {
    "Sunday": 0,
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6
}

matches['home_team_encoded'] = matches['Home'].map(team_code_dict)
matches['away_team_encoded'] = matches['Away'].map(team_code_dict)
matches['day_of_week_encoded'] = matches['day_of_week'].map(day_of_week_code_dict)

# Print first few rows
print(matches.head())

# Define predictors and target
predictors = ['home_team_encoded', 'away_team_encoded', 'day_of_week_encoded', 'Season_End_Year']
target = 'result'

# Initialize Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=1)

# Split data into train and test sets
# train = matches[matches['Date'] < '2022-01-01']
# test = matches[matches['Date'] >= '2022-01-01']

# full train:
train = matches

# Fit the model
rf.fit(train[predictors], train[target])

# Save the trained model to a .pkl file
joblib.dump(rf, 'trained_model4.pkl')

# Make predictions
# predictions = rf.predict(test[predictors])

# Calculate accuracy
# acc = accuracy_score(test[target], predictions)
# print('Accuracy:', acc)

# Calculate precision
# precision = precision_score(test[target], predictions, average='macro')
# print('Precision:', precision)


"""
# Test prediction for an upcoming match
upcoming_match = {
    'Home': 'Wolves',  # Replace with actual home team
    'Away': 'Manchester City',  # Replace with actual away team
    'Date': '2024-10-15'  # Replace with the actual match date
}

# Create a DataFrame for the upcoming match
upcoming_df = pd.DataFrame([upcoming_match])

# Convert Date to datetime and extract day of the week
upcoming_df['Date'] = pd.to_datetime(upcoming_df['Date'])
upcoming_df['day_of_week'] = upcoming_df['Date'].dt.day_name()

# Map teams and day of the week to numerical values using the predefined dictionaries
upcoming_df['home_team_encoded'] = upcoming_df['Home'].map(team_code_dict)
upcoming_df['away_team_encoded'] = upcoming_df['Away'].map(team_code_dict)
upcoming_df['day_of_week_encoded'] = upcoming_df['day_of_week'].map(day_of_week_code_dict)

# Add the Season_End_Year (assuming the match is in the 2024 season)
upcoming_df['Season_End_Year'] = upcoming_df['Date'].dt.year

# Select predictors in the same order as used in training
input_data = upcoming_df[predictors]

# Make the prediction
predicted_result = rf.predict(input_data)

# Map the predicted result back to the original result labels
def reverse_map_ftr(prediction):
    if prediction == 1:
        return 'H'  # Home win
    elif prediction == 0:
        return 'A'  # Away win
    else:
        return 'D'  # Draw

# Convert numeric prediction back to categorical
predicted_result_label = reverse_map_ftr(predicted_result[0])

print('Predicted Result:', predicted_result_label)
"""