import os
import numpy as np
import pickle

BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE, "label_encoders.pkl"), "rb") as f:
    enc = pickle.load(f)

le_team = enc["team"]
le_venue = enc["venue"]

def predict_winner(team1, team2, venue, toss_winner, total_runs, wickets, runrate,
                   boundaries=0, balls_played=120, margin_runs=0, margin_wkts=0, venue_bias=0.5):

    t1 = le_team.transform([team1])[0]
    t2 = le_team.transform([team2])[0]
    venue_enc = le_venue.transform([venue])[0]
    toss_flag = 1 if toss_winner == team1 else 0

    x = np.array([[
        t1, t2, venue_enc, toss_flag,
        total_runs, wickets, runrate,
        boundaries, balls_played,
        margin_runs, margin_wkts,
        venue_bias
    ]])

    prob = model.predict_proba(x)[0]
    winner = le_team.inverse_transform([np.argmax(prob)])[0]
    return winner, prob.tolist()
