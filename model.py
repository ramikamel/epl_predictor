import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib  # Use joblib to load your trained model

# Load your model (make sure you save it after training)
rf = joblib.load('trained_model4.pkl')  # Change to your model file path

# Function to predict match result
def predict_match_result(home_team, away_team, match_date):
    # Create a DataFrame for the input
    upcoming_match = {
        'Home': home_team,
        'Away': away_team,
        'Date': match_date
    }

    upcoming_df = pd.DataFrame([upcoming_match])
    upcoming_df['Date'] = pd.to_datetime(upcoming_df['Date'])
    upcoming_df['day_of_week'] = upcoming_df['Date'].dt.day_name()
    
    # Map the teams and day of week to numerical values
        
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
    
    upcoming_df['home_team_encoded'] = upcoming_df['Home'].map(team_code_dict)
    upcoming_df['away_team_encoded'] = upcoming_df['Away'].map(team_code_dict)
    upcoming_df['day_of_week_encoded'] = upcoming_df['day_of_week'].map(day_of_week_code_dict)
    upcoming_df['Season_End_Year'] = upcoming_df['Date'].dt.year
    
    # Select predictors
    predictors = ['home_team_encoded', 'away_team_encoded', 'day_of_week_encoded', 'Season_End_Year']
    input_data = upcoming_df[predictors]
    
    # Make the prediction
    predicted_result = rf.predict(input_data)

    # Determine the winning team or draw
    if predicted_result == 1:
        return f"{home_team} (Home Win)"
    elif predicted_result == 0:
        return f"{away_team} (Away Win)"
    else:
        return "Draw"

