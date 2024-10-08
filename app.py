from flask import Flask, render_template, request
from model import predict_match_result

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    teams = ["Liverpool", "Arsenal", "Manchester City", "Manchester United", "Chelsea", "Aston Villa", "Brighton", "Newcastle", "Fulham", "Tottenham", "Nott'ham Forest", "Brentford", "West Ham", "Bournemouth", "Leicester City", "Everton", "Ipswich Town", "Crystal Palace", "Southampton", "Wolves"]  # List of 2024 teams

    result = None
    if request.method == 'POST':
        home_team = request.form['home_team']
        away_team = request.form['away_team']
        match_date = request.form['match_date']
        
        result = predict_match_result(home_team, away_team, match_date)

    return render_template('index.html', result=result, teams=teams)


if __name__ == '__main__':
    app.run(debug=True)
