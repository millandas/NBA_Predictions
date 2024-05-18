import pandas as pd
import numpy as np
from scipy.stats import poisson

# Step 1: Load the data
data = pd.read_csv('games.csv')

# Step 2: Calculate average goals scored and conceded by each team at home and away
home_goals_scored_avg = data.groupby('HOME_TEAM_ID')['PTS_home'].mean()
home_goals_conceded_avg = data.groupby('HOME_TEAM_ID')['PTS_away'].mean()
away_goals_scored_avg = data.groupby('VISITOR_TEAM_ID')['PTS_away'].mean()
away_goals_conceded_avg = data.groupby('VISITOR_TEAM_ID')['PTS_home'].mean()

# Function to get the expected goals for home and away teams
def expected_goals(home_team, away_team):
    lambda_home = home_goals_scored_avg[home_team] * away_goals_conceded_avg[away_team] / home_goals_conceded_avg[home_team]
    lambda_away = away_goals_scored_avg[away_team] * home_goals_conceded_avg[home_team] / away_goals_scored_avg[home_team]
    return lambda_home, lambda_away

# Step 4: Use the Poisson distribution
def poisson_prob(lam, k):
    return poisson.pmf(k, lam)

# Function to calculate the probability of different scorelines
def match_probability(home_team, away_team):
    lambda_home, lambda_away = expected_goals(home_team, away_team)
    
    max_goals = 150  # Reasonable upper limit for goals in a match
    home_probs = [poisson_prob(lambda_home, i) for i in range(max_goals)]
    away_probs = [poisson_prob(lambda_away, i) for i in range(max_goals)]
    
    # Create a matrix of probabilities for all scorelines
    prob_matrix = np.outer(home_probs, away_probs)
    
    home_win_prob = np.sum(np.tril(prob_matrix, -1))
    draw_prob = np.sum(np.diag(prob_matrix))
    away_win_prob = np.sum(np.triu(prob_matrix, 1))
    
    return home_win_prob, draw_prob, away_win_prob

# Step 5: Predict match outcomes
def predict_outcome(home_team, away_team):
    home_win_prob, draw_prob, away_win_prob = match_probability(home_team, away_team)
    
    if home_win_prob > draw_prob and home_win_prob > away_win_prob:
        prediction = 'Home Win'
    elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
        prediction = 'Away Win'
    else:
        prediction = 'Draw'
    
    return prediction, home_win_prob, draw_prob, away_win_prob

# Example usage
home_team,away_team = 1610612758,1610612756  # Example team ID for home team
  # Example team ID for away team

prediction, home_win_prob, draw_prob, away_win_prob = predict_outcome(home_team, away_team)

print(f"Prediction: {prediction}")
print(f"Home Win Probability: {home_win_prob:.2f}")
print(f"Draw Probability: {draw_prob:.2f}")
print(f"Away Win Probability: {away_win_prob:.2f}")
